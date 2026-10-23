import { Injectable } from '@nestjs/common';
import { BinaryField, DefaultReadTaskOptions, ExifTool, ReadTaskOptions, Tags, exiftoolPath } from 'exiftool-vendored';
import geotz from 'geo-tz';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { LoggingRepository } from 'src/repositories/logging.repository.js';
import { mimeTypes } from 'src/utils/mime-types.js';

const execFileAsync = promisify(execFile);

/**
 * The trailer read's tags over the base read. The base read never sees the trailer's GPS, so it can only
 * guess the zone (exiftool-vendored's `defaultVideosToUTC`); infer it from the GPS like a normal read would.
 */
export const mergeInsta360TrailerTags = (base: ImmichTags, trailer: ImmichTags): ImmichTags => {
  const tags = { ...base, ...trailer };
  const latitude = Number(trailer.GPSLatitude);
  const longitude = Number(trailer.GPSLongitude);
  const zoneIsGuess = !base.zone || base.zoneSource === 'defaultVideosToUTC';
  if (zoneIsGuess && Number.isFinite(latitude) && Number.isFinite(longitude) && (latitude !== 0 || longitude !== 0)) {
    try {
      // eslint-disable-next-line import-x/no-named-as-default-member
      const zone = geotz.find(latitude, longitude)[0];
      if (zone) {
        Object.assign(tags, { zone, tz: zone, zoneSource: 'GPSLatitude/GPSLongitude' });
      }
    } catch {
      // coordinates geo-tz rejects: keep the base read's zone
    }
  }
  return tags;
};

interface ExifDuration {
  Value: number;
  Scale?: number;
}

type StringOrNumber = string | number;

type TagsWithWrongTypes =
  | 'FocalLength'
  | 'Duration'
  | 'Description'
  | 'ImageDescription'
  | 'RegionInfo'
  | 'TagsList'
  | 'Keywords'
  | 'HierarchicalSubject'
  | 'ISO'
  | 'LensModel';

export interface ImmichTags extends Omit<Tags, TagsWithWrongTypes> {
  ContentIdentifier?: string;
  MotionPhoto?: number;
  MotionPhotoVersion?: number;
  MotionPhotoPresentationTimestampUs?: number;
  MediaGroupUUID?: string;
  ImagePixelDepth?: string;
  FocalLength?: number;
  Duration?: number | string | ExifDuration;
  EmbeddedVideoType?: string;
  EmbeddedVideoFile?: BinaryField;
  MotionPhotoVideo?: BinaryField;
  TagsList?: StringOrNumber[];
  HierarchicalSubject?: StringOrNumber[];
  Keywords?: StringOrNumber | StringOrNumber[];
  ISO?: number | number[];

  // Type is wrong, can also be number.
  Description?: StringOrNumber;
  ImageDescription?: StringOrNumber;

  // Apparently LensModel can also be a float: https://github.com/immich-app/immich/issues/30492
  LensModel?: StringOrNumber;

  // Extended properties for image regions, such as faces
  RegionInfo?: {
    AppliedToDimensions: {
      W: number;
      H: number;
      Unit: string;
    };
    RegionList: {
      Area: {
        // (X,Y) // center of the rectangle
        X: number | string;
        Y: number | string;
        W: number | string;
        H: number | string;
        Unit: string;
      };
      Rotation?: number;
      Type?: string;
      Name?: string;
    }[];
  };

  Device?: {
    Manufacturer?: string;
    ModelName?: string;
  };

  AndroidMake?: string;
  AndroidModel?: string;
  DeviceManufacturer?: string;
  DeviceModelName?: string;

  // Samsung specific tags
  Author?: string;
  SamsungModel?: string;
}

/** A factory so per-call reads can extend the default `readArgs` (a per-call `readArgs` replaces them). */
function createExifToolOptions() {
  return {
    defaultVideosToUTC: true,
    backfillTimezones: true,
    inferTimezoneFromDatestamps: true,
    inferTimezoneFromTimeStamp: true,
    useMWG: true,
    numericTags: [...DefaultReadTaskOptions.numericTags, 'FocalLength', 'FileSize', 'Rotation'],
    /* eslint unicorn/no-array-callback-reference: off, unicorn/no-array-method-this-argument: off */
    // eslint-disable-next-line import-x/no-named-as-default-member
    geoTz: (lat: number, lon: number) => geotz.find(lat, lon)[0],
    geolocation: true,
    readArgs: [
      // Enable exiftool LFS to parse metadata for files larger than 2GB.
      '-api',
      'largefilesupport=1',
      '--ICC_Profile:DeviceManufacturer',
      '--ICC_Profile:DeviceModelName',
      // Ignore embedded thumbnail dimensions/orientation for the main asset.
      '--IFD1:Orientation',
      '--MWG:Orientation',
      '--IFD1:ImageWidth',
      '--IFD1:ImageHeight',
      '--Samsung:Rotation',
    ],
    writeArgs: ['-api', 'largefilesupport=1', '-overwrite_original'],
    taskTimeoutMillis: 2 * 60 * 1000,
  };
}

@Injectable()
export class MetadataRepository {
  private exiftool = new ExifTool(createExifToolOptions());

  constructor(private logger: LoggingRepository) {
    this.logger.setContext(MetadataRepository.name);
  }

  setMaxConcurrency(concurrency: number) {
    this.exiftool.batchCluster.setMaxProcs(concurrency);
  }

  async teardown() {
    await this.exiftool.end();
  }

  async readTags(path: string): Promise<ImmichTags> {
    if (mimeTypes.isInsta360(path)) {
      return this.readInsta360Tags(path);
    }

    const options: ReadTaskOptions | undefined = mimeTypes.isVideo(path) ? { readArgs: ['-ee'] } : undefined;

    return this.exiftool.read(path, options).catch((error) => {
      this.logger.warn(`Error reading exif data (${path}): ${error}\n${error?.stack}`);
      return {};
    }) as Promise<ImmichTags>;
  }

  /**
   * Insta360's raw dual-fisheye files (.insp/.insv) embed camera info (model, firmware,
   * serial number, stitching calibration) in a proprietary trailer, which requires `-ee`
   * to read. For photos the trailer is small and reads fine through the normal shared
   * exiftool process. For videos, the trailer holds per-frame IMU telemetry for the
   * entire recording, and reading it with `-ee` through the long-lived, persistent
   * (`-stay_open`) exiftool process has been observed to intermittently take 60+ seconds
   * and blow past `taskTimeoutMillis` on multi-GB videos - especially under concurrent
   * load - even though a plain, non-persistent `exiftool ... file.insv` invocation handles
   * the exact same file in a few seconds, every time. The slowdown is specific to the
   * persistent-process protocol itself (reproduced even with a freshly-spawned, unshared
   * persistent process), not to which tags are requested, so the fix is to genuinely not
   * use `-stay_open` for this one read - not something exiftool-vendored's `ExifTool`
   * class supports directly, since it always runs in persistent mode.
   */
  private async readInsta360Tags(path: string): Promise<ImmichTags> {
    if (!mimeTypes.isVideo(path)) {
      // .insp: the trailer is small (a single frame's worth of context), so this is fast
      // even through the shared persistent process.
      return this.exiftool.read(path, { readArgs: [...createExifToolOptions().readArgs, '-ee'] }).catch((error) => {
        this.logger.warn(`Error reading exif data (${path}): ${error}\n${(error as Error)?.stack}`);
        return {};
      }) as Promise<ImmichTags>;
    }

    // .insv: standard video metadata (dimensions, dates, ...) reads fine through the shared
    // process without `-ee` - it's specifically the dense per-frame telemetry trailer that's
    // slow. Read the two separately and merge, so a slow/failed trailer read never blocks
    // (or is blocked by) the asset's regular metadata.
    const [baseTags, insta360Fields] = await Promise.all([
      this.exiftool.read(path).catch((error) => {
        this.logger.warn(`Error reading exif data (${path}): ${error}\n${(error as Error)?.stack}`);
        return {};
      }),
      this.readInsta360TrailerFields(path),
    ]);

    return mergeInsta360TrailerTags(baseTags as ImmichTags, insta360Fields);
  }

  /**
   * Reads the trailer-only tags a normal video gets from `-ee` (camera info, GPS) via a genuine
   * one-shot (non-`-stay_open`) exiftool process. Tags the file lacks are simply absent, so they
   * never overwrite the base read.
   */
  private async readInsta360TrailerFields(path: string): Promise<ImmichTags> {
    const tags = ['-Make', '-Model', '-Firmware', '-GPSLatitude', '-GPSLongitude', '-GPSAltitude'];
    const args = ['-api', 'largefilesupport=1', '-ee', '-n', ...tags, '-json', path];

    try {
      const bin = await exiftoolPath();
      const { stdout } = await execFileAsync(bin, args, { maxBuffer: 10 * 1024 * 1024, timeout: 2 * 60 * 1000 });
      const [{ SourceFile: _, ...found }] = JSON.parse(stdout) as [ImmichTags];
      return found;
    } catch (error) {
      this.logger.warn(`Error reading Insta360 trailer fields (${path}): ${error}\n${(error as Error)?.stack}`);
      return {};
    }
  }

  extractBinaryTag(path: string, tagName: string): Promise<Buffer> {
    return this.exiftool.extractBinaryTagToBuffer(tagName, path);
  }

  async writeTags(path: string, tags: Partial<Tags>): Promise<void> {
    // If exiftool assigns a field with ^= instead of =, empty values will be written too.
    // Since exiftool-vendored doesn't support an option for this, we append the ^ to the name of the tag instead.
    // https://exiftool.org/exiftool_pod.html#:~:text=is%20used%20to%20write%20an%20empty%20string
    const tagsToWrite = Object.fromEntries(Object.entries(tags).map(([key, value]) => [`${key}^`, value]));
    try {
      await this.exiftool.write(path, tagsToWrite);
    } catch (error) {
      this.logger.warn(`Error writing exif data (${path}): ${error}`);
    }
  }
}
