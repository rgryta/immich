import { Injectable } from '@nestjs/common';
import { ExifDateTime, WriteTags, exiftool } from 'exiftool-vendored';
import ffmpeg, { FfprobeData, FfprobeStream } from 'fluent-ffmpeg';
import { camelCase, upperFirst } from 'lodash-es';
import { Duration } from 'luxon';
import { execFile, spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Writable } from 'node:stream';
import { promisify } from 'node:util';
import sharp, { Sharp } from 'sharp';
import type {
  Bitmap,
  DecodeToBufferOptions,
  GenerateThumbhashOptions,
  GenerateThumbnailOptions,
  ImageDimensions,
  ProbeOptions,
  TranscodeCommand,
  TransformOptions,
  VideoInfo,
  VideoPacketInfo,
} from 'src/types.js';
import { ORIENTATION_TO_SHARP_ROTATION } from 'src/constants.js';
import { Exif } from 'src/database.js';
import { AssetEditActionItem } from 'src/dtos/editing.dto.js';
import {
  AacProfile,
  Av1Profile,
  ColorMatrix,
  ColorPrimaries,
  ColorTransfer,
  Colorspace,
  DvProfile,
  DvSignalCompatibility,
  H264Profile,
  HevcProfile,
  LogLevel,
  RawExtractedFormat,
} from 'src/enum.js';
import { LoggingRepository } from 'src/repositories/logging.repository.js';
import {
  INSTA360_TAIL_LENGTH,
  Insta360UnsupportedError,
  blendWeight,
  forEachInsta360Row,
  insta360Lenses,
  insta360Orientation,
  insta360PreviewRgb,
  insta360TrailerLength,
  parseInsta360Trailer,
} from 'src/utils/insta360.js';
import { mimeTypes } from 'src/utils/mime-types.js';
import { handlePromiseError } from 'src/utils/misc.js';
import { createAffineMatrix } from 'src/utils/transform.js';

const probe = (input: string, options: string[]): Promise<FfprobeData> =>
  new Promise((resolve, reject) =>
    // eslint-disable-next-line import-x/no-named-as-default-member
    ffmpeg.ffprobe(input, options, (error, data) => (error ? reject(error) : resolve(data))),
  );

const execFileAsync = promisify(execFile);

const assertInsta360PhotoLayout = (width: number, height: number) => {
  if (width !== 2 * height) {
    throw new Insta360UnsupportedError(`unexpected .insp layout ${width}x${height}`);
  }
};

const pascalCase = (str: string) => upperFirst(camelCase(str.toLowerCase()));

type ProgressEvent = {
  frames: number;
  currentFps: number;
  currentKbps: number;
  targetSize: number;
  timemark: string;
  percent?: number;
};

export type ExtractResult = {
  buffer: Buffer;
  format: RawExtractedFormat;
};

@Injectable()
export class MediaRepository {
  constructor(private logger: LoggingRepository) {
    this.logger.setContext(MediaRepository.name);
    // eslint-disable-next-line import-x/no-named-as-default-member
    sharp.concurrency(0);
    // eslint-disable-next-line import-x/no-named-as-default-member
    sharp.cache({ files: 0 });
  }

  /**
   *
   * @param input file path to the input image
   * @returns ExtractResult if succeeded, or null if failed
   */
  async extract(input: string): Promise<ExtractResult | null> {
    if (mimeTypes.isInsta360(input)) {
      // Insta360's raw dual-fisheye photos have both fisheye circles in one JPEG -
      // stitch them into a full-resolution equirectangular photo (see stitchInsta360Photo).
      try {
        const buffer = await this.stitchInsta360Photo(input);
        this.logger.debug(`Successfully stitched Insta360 photo`);
        return { buffer, format: RawExtractedFormat.Jpeg };
      } catch (error: any) {
        this.logger.warn(`Could not stitch Insta360 photo ${input}, falling back to its embedded preview: ${error}`);
      }

      // the camera's own small stitched preview from the trailer
      try {
        return { buffer: await this.extractInsta360Preview(input), format: RawExtractedFormat.Jpeg };
      } catch (error: any) {
        this.logger.warn(`Could not read the Insta360 preview of ${input}: ${error}`);
      }
    }

    for (const { tag, format } of [
      { tag: 'JpgFromRaw2', format: RawExtractedFormat.Jpeg },
      { tag: 'JpgFromRaw', format: RawExtractedFormat.Jpeg },
      { tag: 'PreviewJXL', format: RawExtractedFormat.Jxl },
      { tag: 'PreviewImage', format: RawExtractedFormat.Jpeg },
    ]) {
      try {
        const buffer = await exiftool.extractBinaryTagToBuffer(tag, input);
        this.logger.debug(`Successfully extracted ${tag} buffer from image`);
        return { buffer, format };
      } catch (error: any) {
        this.logger.debug(`Could not extract ${tag} buffer from image: ${error}`);
      }
    }

    return null;
  }

  async writeExif(tags: Partial<Exif>, output: string): Promise<boolean> {
    try {
      const tagsToWrite: WriteTags = {
        ExifImageWidth: tags.exifImageWidth,
        ExifImageHeight: tags.exifImageHeight,
        DateTimeOriginal: tags.dateTimeOriginal && ExifDateTime.fromMillis(tags.dateTimeOriginal.getTime()),
        ModifyDate: tags.modifyDate && ExifDateTime.fromMillis(tags.modifyDate.getTime()),
        TimeZone: tags.timeZone,
        GPSLatitude: tags.latitude,
        GPSLongitude: tags.longitude,
        ProjectionType: tags.projectionType,
        City: tags.city,
        Country: tags.country,
        Make: tags.make,
        Model: tags.model,
        LensModel: tags.lensModel,
        Fnumber: tags.fNumber?.toFixed(1),
        FocalLength: tags.focalLength?.toFixed(1),
        ISO: tags.iso,
        ExposureTime: tags.exposureTime,
        ProfileDescription: tags.profileDescription,
        ColorSpace: tags.colorspace,
        Rating: tags.rating === null ? 0 : tags.rating,
        // specially convert Orientation to numeric Orientation# for exiftool
        'Orientation#': tags.orientation ? Number(tags.orientation) : undefined,
      };

      await exiftool.write(output, tagsToWrite, {
        ignoreMinorErrors: true,
        writeArgs: ['-overwrite_original'],
      });
      return true;
    } catch (error: any) {
      this.logger.warn(`Could not write exif data to image: ${error.message}`);
      return false;
    }
  }

  async copyTagGroup(tagGroup: string, source: string, target: string): Promise<boolean> {
    try {
      await exiftool.write(
        target,
        {},
        {
          ignoreMinorErrors: true,
          writeArgs: ['-TagsFromFile', source, `-${tagGroup}:all>${tagGroup}:all`, '-overwrite_original'],
        },
      );
      return true;
    } catch (error: any) {
      this.logger.warn(`Could not copy tag data to image: ${error.message}`);
      return false;
    }
  }

  async decodeImage(input: string | Buffer, options: DecodeToBufferOptions): Promise<Bitmap> {
    const decoded = await this.getImageDecodingPipeline(input, options).raw().toBuffer({ resolveWithObject: true });
    return await this.transform(decoded, options);
  }

  private edit(pipeline: Sharp, edits: AssetEditActionItem[]): Sharp {
    const crop = edits.find((edit) => edit.action === 'crop');
    if (crop) {
      pipeline = pipeline.extract({
        left: Math.round(crop.parameters.x),
        top: Math.round(crop.parameters.y),
        width: Math.round(crop.parameters.width),
        height: Math.round(crop.parameters.height),
      });
    }

    const affineEditOperations = edits.filter((edit) => edit.action !== 'crop');
    if (affineEditOperations.length > 0) {
      const { a, b, c, d } = createAffineMatrix(affineEditOperations);
      pipeline = pipeline.affine([
        [a, b],
        [c, d],
      ]);
    }

    return pipeline;
  }

  async generateThumbnail(image: Bitmap, options: GenerateThumbnailOptions, output: string): Promise<void> {
    const transformed = await this.transform(image, options);
    await this.tag(transformed, options.colorspace)
      .toFormat(options.format, {
        quality: options.quality,
        // this is default in libvips (except the threshold is 90), but we need to set it manually in sharp
        chromaSubsampling: options.quality >= 80 ? '4:4:4' : '4:2:0',
        progressive: options.progressive,
      })
      .toFile(output);
  }

  private getImageDecodingPipeline(input: string | Buffer, options: DecodeToBufferOptions) {
    // some invalid images can still be processed by sharp, but we want to fail on them by default to avoid crashes
    let pipeline = this.encoded(input, options.processInvalidImages ? 'none' : 'error')
      .pipelineColorspace(this.getPipelineColorspace(options.colorspace))
      .withIccProfile(options.colorspace);

    const { angle, flip, flop } = options.orientation ? ORIENTATION_TO_SHARP_ROTATION[options.orientation] : {};
    pipeline = pipeline.rotate(angle);
    if (flip) {
      pipeline = pipeline.flip();
    }

    if (flop) {
      pipeline = pipeline.flop();
    }

    return pipeline;
  }

  private getPipelineColorspace(colorspace: string) {
    return colorspace === Colorspace.Srgb ? 'srgb' : 'rgb16';
  }

  /* Resamples in linear light; averaging gamma-encoded values darkens the result and loses detail. `colorspace`
   * always has a sRGB transfer function and scRGB applies no primaries matrix, so linearising as sRGB is exact. */
  private transform(image: Bitmap, { size, fit = 'outside', edits = [] }: TransformOptions): Promise<Bitmap> {
    if (!size && edits.length === 0) {
      return Promise.resolve(image);
    }

    return this.edit(this.raw(image).pipelineColorspace('scrgb'), edits)
      .resize(size, size, { fit, withoutEnlargement: true })
      .raw()
      .toBuffer({ resolveWithObject: true });
  }

  private raw({ data, info: raw }: Bitmap) {
    return sharp(data, { raw, limitInputChannels: false, limitInputPixels: false, unlimited: true });
  }

  private encoded(image: string | Buffer, failOn: 'none' | 'error') {
    return sharp(image, { failOn, limitInputChannels: false, limitInputPixels: false, unlimited: true });
  }

  /** Re-attaches the profile, converting nothing: the pixels are already in that colourspace. */
  private tag(image: Bitmap, colorspace: string): Sharp {
    return this.raw(image).pipelineColorspace(this.getPipelineColorspace(colorspace)).withIccProfile(colorspace);
  }

  async generateThumbhash(image: Bitmap, options: GenerateThumbhashOptions): Promise<Buffer> {
    const { rgbaToThumbHash } = await import('thumbhash');

    const transformed = await this.transform(image, { edits: options.edits, fit: 'inside', size: 100 });

    const { data, info } = await sharp(transformed.data, { raw: transformed.info })
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    return Buffer.from(rgbaToThumbHash(info.width, info.height, data));
  }

  async probe(input: string, options?: ProbeOptions): Promise<VideoInfo> {
    const results = await probe(input, options?.countFrames ? ['-count_packets'] : []); // gets frame count quickly: https://stackoverflow.com/a/28376817
    return {
      format: {
        formatName: results.format.format_name,
        formatLongName: results.format.format_long_name,
        duration: this.parseFloat(results.format.duration),
        bitrate: this.parseInt(results.format.bit_rate),
      },
      videoStreams: results.streams
        .filter((stream) => stream.codec_type === 'video' && !stream.disposition?.attached_pic)
        .sort((a, b) => this.compareStreams(a, b))
        .map((stream) => {
          const height = this.parseInt(stream.height);
          const dar = this.getDar(stream.display_aspect_ratio);
          return {
            index: stream.index,
            height,
            width: dar ? Math.round(height * dar) : this.parseInt(stream.width),
            codecName: stream.codec_name === 'h265' ? 'hevc' : (stream.codec_name ?? null),
            profile: this.parseVideoProfile(stream.codec_name, stream.profile as string | undefined) ?? null,
            level: this.parseOptionalInt(stream.level),
            frameCount: this.parseInt(options?.countFrames ? stream.nb_read_packets : stream.nb_frames),
            frameRate: this.parseFrameRate(stream.avg_frame_rate ?? stream.r_frame_rate),
            timeBase: this.parseRational(stream.time_base)?.den ?? null,
            rotation: this.parseInt(stream.rotation),
            bitrate: this.parseInt(stream.bit_rate),
            pixelFormat: stream.pix_fmt || 'yuv420p',
            colorPrimaries: this.parseEnum(ColorPrimaries, stream.color_primaries) ?? ColorPrimaries.Unknown,
            colorMatrix: this.parseEnum(ColorMatrix, stream.color_space) ?? ColorMatrix.Unknown,
            colorTransfer: this.parseEnum(ColorTransfer, stream.color_transfer) ?? ColorTransfer.Unknown,
            dvProfile: this.parseOptionalInt(stream.dv_profile) as DvProfile | null,
            dvLevel: this.parseOptionalInt(stream.dv_level),
            dvBlSignalCompatibilityId: this.parseOptionalInt(
              stream.dv_bl_signal_compatibility_id,
            ) as DvSignalCompatibility | null,
          };
        }),
      audioStreams: results.streams
        .filter((stream) => stream.codec_type === 'audio' && stream.codec_name !== 'unknown')
        .sort((a, b) => this.compareStreams(a, b))
        .map((stream) => ({
          index: stream.index,
          codecName: stream.codec_name ?? null,
          profile:
            stream.codec_name === 'aac' ? this.parseEnum(AacProfile, stream.profile as string | undefined) : null,
          bitrate: this.parseInt(stream.bit_rate),
        })),
    };
  }

  /**
   * Needed for accurate segments, especially when remuxing, seeking and/or VFR is involved.
   * Scanning packets for keyframes in JS is much faster than -skip_frame nokey since it avoids decoding the video.
   */
  probePackets(input: string, streamIndex: number): Promise<VideoPacketInfo | null> {
    const ffprobe = spawn(
      'ffprobe',
      [
        '-v',
        'error',
        '-select_streams',
        String(streamIndex),
        '-show_entries',
        'packet=pts,duration,flags',
        '-of',
        'csv=p=0',
        input,
      ],
      { stdio: ['ignore', 'pipe', 'pipe'] },
    );

    let totalDuration = 0;
    const keyframePts: number[] = [];
    const keyframeAccDuration: number[] = [];
    const keyframeOwnDuration: number[] = [];
    const postDiscard: { pts: number; duration: number }[] = [];
    const parseLine = (line: string) => {
      if (!line) {
        return;
      }
      const [ptsStr, durationStr, flags] = line.split(',', 3);
      const pts = Number.parseInt(ptsStr);
      const duration = Number.parseInt(durationStr);
      if (Number.isNaN(pts) || Number.isNaN(duration) || !flags) {
        return;
      }
      // Discarded packets don't contribute to packet count, but still contribute to video duration
      totalDuration += duration;
      if (flags[1] !== 'D') {
        postDiscard.push({ pts, duration });
      }

      if (flags[0] !== 'K') {
        return;
      }

      keyframePts.push(pts);
      keyframeAccDuration.push(totalDuration);
      // VFR content can have variable duration keyframes,
      // so we need to track their duration separately for accurate segment boundaries.
      // Non-keyframes are accounted for in totalDuration.
      keyframeOwnDuration.push(duration);
    };

    let stderr = '';
    let remainder = '';
    ffprobe.stderr.setEncoding('utf8');
    ffprobe.stderr.on('data', (chunk: string) => (stderr += chunk));
    ffprobe.stdout.setEncoding('utf8');
    ffprobe.stdout.on('data', (chunk: string) => {
      const lines = chunk.split('\n');
      lines[0] = remainder + lines[0];
      remainder = lines.pop() as string;
      for (const line of lines) {
        parseLine(line);
      }
    });

    return new Promise<VideoPacketInfo | null>((resolve, reject) => {
      ffprobe.on('error', reject);
      ffprobe.on('close', (code) => {
        if (code !== 0) {
          return reject(new Error(`ffprobe exited with code ${code}: ${stderr.trim()}`));
        }
        parseLine(remainder);
        if (postDiscard.length === 0) {
          return resolve(null);
        }

        resolve({
          totalDuration,
          packetCount: postDiscard.length,
          outputFrames: this.cfrOutputFrames(postDiscard, postDiscard.length / totalDuration),
          keyframePts,
          keyframeAccDuration,
          keyframeOwnDuration,
        });
      });
    });
  }

  transcode(input: string, output: string | Writable, options: TranscodeCommand): Promise<void> {
    if (!options.twoPass) {
      return new Promise((resolve, reject) => {
        this.configureFfmpegCall(input, output, options)
          .on('error', reject)
          .on('end', () => resolve())
          .run();
      });
    }

    if (typeof output !== 'string') {
      throw new TypeError('Two-pass transcoding does not support writing to a stream');
    }

    // two-pass allows for precise control of bitrate at the cost of running twice
    // recommended for vp9 for better quality and compression
    return new Promise((resolve, reject) => {
      // first pass output is not saved as only the .log file is needed
      this.configureFfmpegCall(input, '/dev/null', options)
        .addOptions('-pass', '1')
        .addOptions('-passlogfile', output)
        .addOptions('-f null')
        .on('error', reject)
        .on('end', () => {
          // second pass
          this.configureFfmpegCall(input, output, options)
            .addOptions('-pass', '2')
            .addOptions('-passlogfile', output)
            .on('error', reject)
            .on('end', () => handlePromiseError(fs.unlink(`${output}-0.log`), this.logger))
            .on('end', () => handlePromiseError(fs.rm(`${output}-0.log.mbtree`, { force: true }), this.logger))
            .on('end', () => resolve())
            .run();
        })
        .run();
    });
  }

  /**
   * Stitches an Insta360 X5 raw dual-fisheye video (one HEVC stream per lens) into a single
   * equirectangular MP4 with Immich's own ffmpeg: each lens is remapped through lookup maps
   * computed once from the file's calibration (see utils/insta360.ts), then blended by mask.
   */
  async stitchInsta360Video(input: string, output: string): Promise<void> {
    const trailer = await this.readInsta360Trailer(input);
    const stream = await this.insta360LensStream(input);
    const width = stream.width!;
    const lensHeight = stream.height ?? width;
    const height = width / 2;
    // shortcut: levelled once, at the first frame; per-frame levelling needs per-frame maps (a custom filter)
    const lensData = insta360Lenses(trailer, width, insta360Orientation(trailer));

    // remap is nearest-neighbour; coordinates outside the lens crop become black (65535 = unmapped)
    const xmaps = [new Uint16Array(width * height), new Uint16Array(width * height)];
    const ymaps = [new Uint16Array(width * height), new Uint16Array(width * height)];
    const mask = new Uint8Array(width * height);
    await forEachInsta360Row(width, height, lensData, (y, u0, v0, u1, v1, w0) => {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        const valid = [0, 1].map((l) => {
          const u = Math.round(l ? u1[x] : u0[x]);
          const v = Math.round(l ? v1[x] : v0[x]);
          const ok = u >= 0 && u < width && v >= 0 && v < lensHeight;
          xmaps[l][i] = ok ? u : 65_535;
          ymaps[l][i] = ok ? v : 65_535;
          return ok;
        });
        mask[i] = Math.round(255 * blendWeight(valid[0], valid[1], w0[x]));
      }
    });

    const folder = await fs.mkdtemp(join(tmpdir(), 'insta360-'));
    try {
      const planes = { x0: xmaps[0], y0: ymaps[0], x1: xmaps[1], y1: ymaps[1], mask };
      for (const [name, plane] of Object.entries(planes)) {
        await fs.writeFile(join(folder, name), plane);
      }
      const rawInput = (name: string, format: string) => [
        '-f',
        'rawvideo',
        '-pix_fmt',
        format,
        '-s',
        `${width}x${height}`,
        '-i',
        join(folder, name),
      ];
      await execFileAsync(
        'ffmpeg',
        [
          '-v',
          'error',
          '-y',
          '-i',
          input,
          ...rawInput('x0', 'gray16le'),
          ...rawInput('y0', 'gray16le'),
          ...rawInput('x1', 'gray16le'),
          ...rawInput('y1', 'gray16le'),
          ...rawInput('mask', 'gray'),
          '-filter_complex',
          // .insv stores calibration lens 1 as the first video stream
          '[0:v:1]format=gbrp[l0];[0:v:0]format=gbrp[l1];[5]format=gbrp[m];' +
            '[l0][1][2]remap[f];[l1][3][4]remap[b];[b][f][m]maskedmerge,format=yuv420p[out]',
          '-map',
          '[out]',
          '-map',
          '0:a?',
          '-c:v',
          'libx264',
          '-preset',
          'medium',
          '-crf',
          '20',
          '-c:a',
          'copy',
          '-movflags',
          '+faststart',
          output,
        ],
        { maxBuffer: 10 * 1024 * 1024 },
      );
    } finally {
      await fs.rm(folder, { recursive: true, force: true });
    }
  }

  /**
   * Stitches an Insta360 X5 raw dual-fisheye photo (one JPEG holding both fisheye circles)
   * into a color equirectangular JPEG, the way Insta360 Studio does. Everything needed -
   * lens calibration and the IMU stream Studio levels the photo with - comes from the
   * .insp file's own trailer; see utils/insta360.ts.
   */
  stitchInsta360Photo(input: string): Promise<Buffer> {
    // the stitch is single-threaded JS holding ~1 GB: queueing loses no speed and caps the memory
    const result = this.insta360PhotoStitches.then(() => this.stitchInsta360PhotoNow(input));
    this.insta360PhotoStitches = result.catch(() => {});
    return result;
  }

  private insta360PhotoStitches: Promise<unknown> = Promise.resolve();

  private async stitchInsta360PhotoNow(input: string): Promise<Buffer> {
    const trailer = await this.readInsta360Trailer(input);
    const { data, info } = await sharp(input).raw().toBuffer({ resolveWithObject: true });
    const { width, height, channels } = info;
    assertInsta360PhotoLayout(width, height);
    const lensData = insta360Lenses(trailer, height, insta360Orientation(trailer));
    const out = Buffer.alloc(data.length);
    const a = new Float64Array(channels);
    const b = new Float64Array(channels);
    // neighbours outside a lens crop count as black
    const tap = (lens: number, x: number, y: number, weight: number, into: Float64Array) => {
      if (x < 0 || x >= height || y < 0 || y >= height) {
        return;
      }
      const i = (y * width + x + lens * height) * channels;
      for (let c = 0; c < channels; c++) {
        into[c] += data[i + c] * weight;
      }
    };
    // bilinear like Studio's single texture fetch
    const sample = (lens: number, u: number, v: number, into: Float64Array) => {
      into.fill(0);
      if (Number.isNaN(u)) {
        return false;
      }
      const x0 = Math.floor(u);
      const y0 = Math.floor(v);
      const fx = u - x0;
      const fy = v - y0;
      tap(lens, x0, y0, (1 - fx) * (1 - fy), into);
      tap(lens, x0 + 1, y0, fx * (1 - fy), into);
      tap(lens, x0, y0 + 1, (1 - fx) * fy, into);
      tap(lens, x0 + 1, y0 + 1, fx * fy, into);
      return true;
    };
    await forEachInsta360Row(width, height, lensData, (y, u0, v0, u1, v1, w0) => {
      for (let x = 0; x < width; x++) {
        const weight = blendWeight(sample(0, u0[x], v0[x], a), sample(1, u1[x], v1[x], b), w0[x]);
        const o = (y * width + x) * channels;
        for (let c = 0; c < channels; c++) {
          out[o + c] = a[c] * weight + b[c] * (1 - weight) + 0.5;
        }
      }
    });
    // quality=100 matches Insta360 Studio's own JPEG output (the flat all-1s IJG table)
    return sharp(out, { raw: { width, height, channels } }).jpeg({ quality: 100 }).toBuffer();
  }

  /** The camera's own small stitched equirectangular frame from the trailer, as JPEG. */
  async extractInsta360Preview(input: string): Promise<Buffer> {
    const { preview } = await this.readInsta360Trailer(input);
    if (!preview) {
      throw new Error('no preview in trailer');
    }
    const { width, height } = preview;
    return sharp(insta360PreviewRgb(preview), { raw: { width, height, channels: 3 } })
      .jpeg()
      .toBuffer();
  }

  /** Whether an Insta360 file can be stitched, and whether its trailer carries the camera's own stitched preview. */
  async probeInsta360(input: string): Promise<{ stitchable: boolean; preview: boolean }> {
    let preview = false;
    try {
      const trailer = await this.readInsta360Trailer(input);
      preview = !!trailer.preview;
      insta360Lenses(trailer, 1, insta360Orientation(trailer));
      // the stitchers' own checks on the media itself, so the probe can never disagree with them
      if (mimeTypes.isVideo(input)) {
        await this.insta360LensStream(input);
      } else {
        const { width = 0, height = 0 } = await sharp(input).metadata();
        assertInsta360PhotoLayout(width, height);
      }
      return { stitchable: true, preview };
    } catch (error) {
      if (error instanceof Insta360UnsupportedError) {
        return { stitchable: false, preview };
      }
      throw error;
    }
  }

  /** The first of the two lens streams a stitchable .insv carries. */
  private async insta360LensStream(input: string) {
    const { streams } = await probe(input, []);
    const [stream, ...others] = streams.filter((stream) => stream.codec_type === 'video');
    if (!stream?.width || others.length === 0) {
      throw new Insta360UnsupportedError('expected one video stream per lens');
    }
    return stream;
  }

  /** Reads only the trailer at the end of the file (videos are several GB). */
  private async readInsta360Trailer(input: string) {
    const file = await fs.open(input, 'r');
    try {
      const { size } = await file.stat();
      const { buffer: tail } = await file.read(
        Buffer.alloc(INSTA360_TAIL_LENGTH),
        0,
        INSTA360_TAIL_LENGTH,
        Math.max(size - INSTA360_TAIL_LENGTH, 0),
      );
      const length = insta360TrailerLength(tail, size);
      const { buffer, bytesRead } = await file.read(Buffer.alloc(length), 0, length, size - length);
      // a short read would parse as "no records" and look unsupported; it is an I/O error worth a retry
      if (bytesRead !== length) {
        throw new Error(`short read of the Insta360 trailer (${bytesRead} of ${length} bytes)`);
      }
      return parseInsta360Trailer(buffer);
    } finally {
      await file.close();
    }
  }

  async getImageMetadata(input: string | Buffer): Promise<ImageDimensions & { isTransparent: boolean }> {
    const { width = 0, height = 0, hasAlpha = false } = await sharp(input, { unlimited: true }).metadata();
    return { width, height, isTransparent: hasAlpha };
  }

  private configureFfmpegCall(input: string, output: string | Writable, options: TranscodeCommand) {
    const ffmpegCall = ffmpeg(input, { niceness: 10 })
      .inputOptions(options.inputOptions)
      .outputOptions(options.outputOptions)
      .output(output)
      .on('start', (command: string) => this.logger.debug(command))
      .on('error', (error, _, stderr) => this.logger.error(stderr || error));

    const { frameCount, percentInterval } = options.progress;
    const frameInterval = Math.ceil(frameCount / (100 / percentInterval));
    if (this.logger.isLevelEnabled(LogLevel.Debug) && frameCount && frameInterval) {
      let lastProgressFrame: number = 0;
      ffmpegCall.on('progress', (progress: ProgressEvent) => {
        if (progress.frames - lastProgressFrame < frameInterval) {
          return;
        }

        lastProgressFrame = progress.frames;
        const percent = ((progress.frames / frameCount) * 100).toFixed(2);
        const ms = progress.currentFps ? Math.floor((frameCount - progress.frames) / progress.currentFps) * 1000 : 0;
        const duration = ms ? Duration.fromMillis(ms).rescale().toHuman({ unitDisplay: 'narrow' }) : '';
        const outputText = output instanceof Writable ? 'stream' : output.split('/').pop();
        this.logger.debug(
          `Transcoding ${percent}% done${duration ? `, estimated ${duration} remaining` : ''} for output ${outputText}`,
        );
      });
    }

    return ffmpegCall;
  }

  private parseInt(value: string | number | undefined): number {
    return Number.parseInt(value as string) || 0;
  }

  private parseFloat(value: string | number | undefined): number {
    // eslint-disable-next-line unicorn/prefer-number-coercion
    return Number.parseFloat(value as string) || 0;
  }

  private parseOptionalInt(value: string | number | undefined): number | null {
    const parsed = Number.parseInt(value as string);
    return Number.isNaN(parsed) ? null : parsed;
  }

  private parseEnum<E extends Record<string, number | string>>(enumObj: E, value?: string) {
    return value && typeof value === 'string'
      ? ((enumObj[pascalCase(value)] as Extract<E[keyof E], number> | undefined) ?? null)
      : null;
  }

  /** Parse a rational like "60000/1001" or "1/600" into `{ num, den }`. */
  private parseRational(value: string | undefined): { num: number; den: number } | null {
    if (value) {
      const [num, den = 1] = value.split('/').map(Number);
      if (num && den) {
        return { num, den };
      }
    }
    return null;
  }

  private parseFrameRate(value: string | undefined): number | null {
    const r = this.parseRational(value);
    return r ? r.num / r.den : null;
  }

  private getDar(dar: string | undefined): number {
    if (dar) {
      const [darW, darH] = dar.split(':').map(Number);
      if (darW && darH) {
        return darW / darH;
      }
    }

    return 0;
  }

  private parseVideoProfile(codec?: string, profile?: string) {
    switch (codec) {
      case 'h264': {
        return this.parseEnum(H264Profile, profile);
      }
      case 'h265':
      case 'hevc': {
        return this.parseEnum(HevcProfile, profile);
      }
      case 'av1': {
        return this.parseEnum(Av1Profile, profile);
      }
      default: {
        return null;
      }
    }
  }

  private compareStreams(a: FfprobeStream, b: FfprobeStream): number {
    const d = (b.disposition?.default ?? 0) - (a.disposition?.default ?? 0);
    if (d !== 0) {
      return d;
    }
    return this.parseInt(b.bit_rate) - this.parseInt(a.bit_rate);
  }

  /* Ported from https://code.ffmpeg.org/FFmpeg/FFmpeg/src/commit/5c44245878e235ae64fe87fb9877644856d33d1d/fftools/ffmpeg_filter.c
   * SPDX-License-Identifier: LGPL-2.1-or-later
   * Copyright (c) FFmpeg authors and contributors — https://ffmpeg.org/
   * Modifications: TS port operating on probe-derived packet metadata rather than decoded AVFrames. */
  private cfrOutputFrames(packets: { pts: number; duration: number }[], slotsPerTick: number) {
    packets.sort((a, b) => a.pts - b.pts);
    const firstPts = packets[0].pts;
    let outputFrames = 0;
    let nextPts = 0;
    const history = [0, 0, 0];
    for (const pkt of packets) {
      const syncIpts = (pkt.pts - firstPts) * slotsPerTick;
      const duration = pkt.duration * slotsPerTick;
      let delta0 = syncIpts - nextPts;
      const delta = delta0 + duration;

      if (delta0 < 0 && delta > 0) {
        delta0 = 0;
      }

      let nb = 1;
      let nbPrev = 0;
      if (delta < -1.1) {
        nb = 0;
      } else if (delta > 1.1) {
        nb = Math.round(delta);
        if (delta0 > 1.1) {
          nbPrev = Math.round(delta0 - 0.6);
        }
      }
      outputFrames += nb;
      nextPts += nb;
      history[2] = history[1];
      history[1] = history[0];
      history[0] = nbPrev;
    }
    const median = history.sort((a, b) => a - b)[1];
    return outputFrames + median;
  }
}
