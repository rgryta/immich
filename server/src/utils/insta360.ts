/**
 * Insta360 X5 dual-fisheye stitching, matched to Insta360 Studio's own native stitcher
 * (libarvbmg.so / studio_worker.dll). Every constant is the value Studio uses, not a fit:
 *
 * - Calibration: ExtraMetadata (trailer record 0x101) field 5 `offset`, laid out like
 *   ins::Offset::parseOffset: count, [radius cx cy rx ry rz] x2, calib_w, calib_h, lensType.
 * - Lens model: ins::Lens::getCoeff(113) quartic angle -> radius map (the shader's calTex).
 *   Per-lens rotation FisheyeImageModel::RotateMatrix = Ry(ry + 90) * Rz(rx) * Rx(rz),
 *   post-multiplied by P = Rx(90) * Ry(90) (lens 0) and Rz(180) * P (lens 1).
 * - Seam: calAlpha with the X5 blend angle from GetLensConfig (18 deg), nadir narrowing and
 *   the alphaNormalized S-curve from the fragment shader.
 * - Orientation: the SequenceStabilizer path Studio uses for photos -- the raw IMU stream
 *   (trailer record 0x300) through GyroProcessImpl's box-gravity filter, sampled at
 *   first_frame_timestamp; buffers shorter than the 10 s box fall back to
 *   SingleGyroStabilizer (minimal rotation of the first sample's gravity).
 */
import { setImmediate } from 'node:timers/promises';

export type Vec3 = [number, number, number];
export type Mat3 = [Vec3, Vec3, Vec3];
type Quat = [number, number, number, number]; // x, y, z, w

const COEFF_BY_LENSTYPE: Record<number, number[]> = {
  113: [0, 0.03159, -8.415e-5, 2.201e-6, -1.284e-8], // X5
};
// Half-FOV by ExtraMetadata field 68 `offset_states` (the OffsetState enum): bare lenses match
// Lens::getFov(113) / 2; the Premium (S) guard is fitted to Studio exports (97.43 / 97.56 / 97.45).
// shortcut: Protect_A, diving housings and ND filters are unmeasured, so they throw until a sample exists.
const HALF_FOV_BY_OFFSET_STATE: Record<number, number> = {
  0: 100, // Common
  5: 97.5, // Protect_S
};
const BLEND_ANGLE_DEG = 18;
const NADIR_BLEND_FLOOR_DEG = 5;
const ALPHA_LAMBDA = 5.19999980926513671875;

const RECORD_DIRECTORY = 0x0;
const RECORD_EXTRA_METADATA = 0x1_01;
const RECORD_PREVIEW = 0x2_00;
const RECORD_IMU = 0x3_00;
const TRAILER_MAGIC = '8db42d694ccc418790edff439fe026bf';
const GRAVITY_BOX_WINDOW_S = 10; // GyroProcessImpl +0x88
const GRAVITY_LOWPASS_HZ = 0.02; // iirf::getButterCoeff(fs, 0.02, 2), then iirf::filtFilt
const QUAT_TIME_OFFSET_S = 0.001; // pushGLMatrix(q, (t_prev + t_cur) / 2 + 0.001)
// IMU decoded past the first frame; the 0.02 Hz filtfilt gives later samples no weight, and a long
// recording's whole IMU stream would otherwise cost ~1 GB of heap per 30 min
const IMU_WINDOW_US = 120e6;

// Signed-permutation frames of Studio's orientation chain: ins::AlignAxes for the X5 gyro
// type, and the sphere <- camera / output <- sphere conventions of the render model.
const IMU_ALIGN: Mat3 = [
  [-1, 0, 0],
  [0, 0, 1],
  [0, 1, 0],
];
const SPHERE_IN: Mat3 = [
  [-1, 0, 0],
  [0, 0, -1],
  [0, -1, 0],
];
const SPHERE_OUT: Mat3 = [
  [1, 0, 0],
  [0, 0, 1],
  [0, -1, 0],
];

/** The file itself cannot be stitched (another camera, accessory or trailer layout), as opposed to an I/O failure. */
export class Insta360UnsupportedError extends Error {}

export type Insta360Trailer = {
  offset: number[];
  offsetState: number;
  firstFrameUs?: number;
  accRange: number;
  gyroRange: number;
  imu?: { ts: Float64Array; raw: Uint16Array }; // raw: 6 per sample, accel xyz then gyro xyz
  preview?: { width: number; height: number; nv12: Buffer }; // the camera's own small stitched frame
};

// ---- trailer --------------------------------------------------------------------------------

export const INSTA360_TAIL_LENGTH = 78;

/** The whole trailer's length, from a file's last 78 bytes; checked before anything that size is read. */
export const insta360TrailerLength = (tail: Buffer, fileSize: number) => {
  const length = tail.length === INSTA360_TAIL_LENGTH ? tail.readUInt32LE(38) : 0;
  if (tail.toString('latin1', 46) !== TRAILER_MAGIC || length < INSTA360_TAIL_LENGTH || length > fileSize) {
    throw new Insta360UnsupportedError('no Insta360 trailer');
  }
  return length;
};

/** Trailer v3: every record's [u16 id][u32 length] header follows its payload; the last header starts the 78-byte tail. */
export const parseInsta360Trailer = (trailer: Buffer): Insta360Trailer => {
  const records = new Map<number, Buffer>();
  for (let header = trailer.length - 78; header >= 6;) {
    const id = trailer.readUInt16LE(header);
    const length = trailer.readUInt32LE(header + 2);
    if (length > header) {
      break;
    }
    if (id === RECORD_DIRECTORY) {
      // .insv: a table of [u16 id][u32 size][u32 offset from trailer start] replaces the walk
      for (let entry = header - length; entry + 10 <= header; entry += 10) {
        const size = trailer.readUInt32LE(entry + 2);
        const offset = trailer.readUInt32LE(entry + 6);
        if (size > 0 && offset + size + 6 <= trailer.length) {
          // the table abbreviates ids (0x3 for 0x300); the record's own footer has the full one
          records.set(trailer.readUInt16LE(offset + size), trailer.subarray(offset, offset + size));
        }
      }
      break;
    }
    records.set(id, trailer.subarray(header - length, header));
    header -= length + 6;
  }

  const metadata = records.get(RECORD_EXTRA_METADATA);
  if (!metadata) {
    throw new Insta360UnsupportedError('no ExtraMetadata record in trailer');
  }
  const result: Insta360Trailer = { offset: [], offsetState: 0, accRange: 8, gyroRange: 2000 };
  for (const { field, value } of protobufFields(metadata)) {
    if (field === 5 && Buffer.isBuffer(value) && result.offset.length === 0) {
      result.offset = value.toString('ascii').split('_').map(Number);
    } else if (field === 24 && typeof value === 'number') {
      result.firstFrameUs = value;
    } else if (field === 68 && typeof value === 'number') {
      result.offsetState = value;
    } else if (field === 65 && Buffer.isBuffer(value)) {
      for (const range of protobufFields(value)) {
        if (range.field === 1 && typeof range.value === 'number') {
          result.accRange = range.value;
        } else if (range.field === 2 && typeof range.value === 'number') {
          result.gyroRange = range.value;
        }
      }
    }
  }
  if (result.offset.length !== 16 || result.offset.some((value) => !Number.isFinite(value))) {
    throw new Insta360UnsupportedError('no ExtraMetadata calibration offset');
  }

  const imu = records.get(RECORD_IMU);
  if (imu && imu.length >= 20) {
    const start = Number(imu.readBigUInt64LE(0));
    const limit = Math.max(result.firstFrameUs ?? start, start) + IMU_WINDOW_US;
    let count = 0;
    while (count < Math.floor(imu.length / 20) && Number(imu.readBigUInt64LE(count * 20)) <= limit) {
      count++;
    }
    const ts = new Float64Array(count);
    const raw = new Uint16Array(count * 6);
    for (let i = 0; i < count; i++) {
      ts[i] = Number(imu.readBigUInt64LE(i * 20));
      for (let k = 0; k < 6; k++) {
        raw[i * 6 + k] = imu.readUInt16LE(i * 20 + 8 + k * 2);
      }
    }
    if (count > 0) {
      result.imu = { ts, raw };
    }
  }

  // [u32 1][u32 record length][u32 1][u32 0][u32 width][u32 height], padded to 40 bytes, then NV12
  const preview = records.get(RECORD_PREVIEW);
  if (preview && preview.length >= 40 && preview.readUInt32LE(0) === 1) {
    const width = preview.readUInt32LE(16);
    const height = preview.readUInt32LE(20);
    if (preview.length === 40 + width * height * 1.5) {
      result.preview = { width, height, nv12: preview.subarray(40) };
    }
  }
  return result;
};

function* protobufFields(buffer: Buffer): Generator<{ field: number; value: number | Buffer }> {
  let i = 0;
  const varint = () => {
    let result = 0;
    let shift = 1;
    while (i < buffer.length) {
      const byte = buffer[i++];
      result += (byte & 0x7f) * shift;
      if (!(byte & 0x80)) {
        return result;
      }
      shift *= 128;
    }
    throw new Insta360UnsupportedError('truncated protobuf');
  };
  while (i < buffer.length) {
    const tag = varint();
    const field = Math.floor(tag / 8);
    const wireType = tag % 8;
    switch (wireType) {
      case 0: {
        yield { field, value: varint() };
        break;
      }
      case 1: {
        i += 8;
        break;
      }
      case 5: {
        i += 4;
        break;
      }
      case 2: {
        const length = varint();
        yield { field, value: buffer.subarray(i, i + length) };
        i += length;
        break;
      }
      default: {
        return;
      }
    }
  }
}

// ---- small linear algebra -------------------------------------------------------------------

const mul = (a: Mat3, b: Mat3): Mat3 =>
  a.map((row) => [0, 1, 2].map((j) => row[0] * b[0][j] + row[1] * b[1][j] + row[2] * b[2][j])) as Mat3;
const transpose = (m: Mat3): Mat3 => [0, 1, 2].map((i) => [m[0][i], m[1][i], m[2][i]]) as Mat3;
const apply = (m: Mat3, v: Vec3): Vec3 => m.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]) as Vec3;
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const normalize = (v: Vec3): Vec3 => {
  const n = Math.hypot(...v);
  return [v[0] / n, v[1] / n, v[2] / n];
};

const axisRotation = (axis: 'x' | 'y' | 'z', degrees: number): Mat3 => {
  const t = (degrees * Math.PI) / 180;
  const c = Math.cos(t);
  const s = Math.sin(t);
  if (axis === 'x') {
    return [
      [1, 0, 0],
      [0, c, -s],
      [0, s, c],
    ];
  }
  if (axis === 'y') {
    return [
      [c, 0, s],
      [0, 1, 0],
      [-s, 0, c],
    ];
  }
  return [
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1],
  ];
};

/** FisheyeImageModel::RotateMatrix, identical to the lens matrix Insta360Lens::OffsetParser stores. */
const rotateMatrix = (rx: number, ry: number, rz: number) =>
  mul(mul(axisRotation('y', ry + 90), axisRotation('z', rx)), axisRotation('x', rz));

/** Minimal rotation taking direction a onto direction b. */
const rotationBetween = (a: Vec3, b: Vec3): Mat3 => {
  const [na, nb] = [normalize(a), normalize(b)];
  const v = cross(na, nb);
  const c = dot(na, nb);
  const s = Math.hypot(...v);
  if (s < 1e-8) {
    return c > 0
      ? [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ]
      : [
          [-1, 0, 0],
          [0, -1, 0],
          [0, 0, -1],
        ];
  }
  const k: Mat3 = [
    [0, -v[2], v[1]],
    [v[2], 0, -v[0]],
    [-v[1], v[0], 0],
  ];
  const k2 = mul(k, k);
  const f = (1 - c) / (s * s);
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === j ? 1 : 0) + k[i][j] + k2[i][j] * f)) as Mat3;
};

const qmul = ([ax, ay, az, aw]: Quat, [bx, by, bz, bw]: Quat): Quat => [
  aw * bx + ax * bw + ay * bz - az * by,
  aw * by - ax * bz + ay * bw + az * bx,
  aw * bz + ax * by - ay * bx + az * bw,
  aw * bw - ax * bx - ay * by - az * bz,
];
const qconj = ([x, y, z, w]: Quat): Quat => [-x, -y, -z, w];
const qnormalize = (q: Quat): Quat => {
  const n = Math.hypot(...q);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
};
export const qrotate = (q: Quat, v: Vec3): Vec3 => {
  const u: Vec3 = [q[0], q[1], q[2]];
  const t = cross(u, v).map((value) => value * 2) as Vec3;
  const ut = cross(u, t);
  return [v[0] + q[3] * t[0] + ut[0], v[1] + q[3] * t[1] + ut[1], v[2] + q[3] * t[2] + ut[2]];
};
const qmatrix = ([x, y, z, w]: Quat): Mat3 => [
  [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
];

// ---- orientation ----------------------------------------------------------------------------

/** 2nd-order Butterworth low-pass, bilinear transform with prewarp (iirf::getButterCoeff). */
export const butter2Lowpass = (fc: number, fs: number) => {
  const k = Math.tan((Math.PI * fc) / fs);
  const n = 1 / (1 + Math.SQRT2 * k + k * k);
  const b0 = k * k * n;
  return { b: [b0, 2 * b0, b0], a: [1, 2 * (k * k - 1) * n, (1 - Math.SQRT2 * k + k * k) * n] };
};

const lfilter2 = (b: number[], a: number[], x: number[], z0: number, z1: number) => {
  const y = Array.from<number>({ length: x.length });
  for (let i = 0; i < x.length; i++) {
    y[i] = b[0] * x[i] + z0;
    z0 = b[1] * x[i] - a[1] * y[i] + z1;
    z1 = b[2] * x[i] - a[2] * y[i];
  }
  return y;
};

/** MATLAB-style filtfilt (iirf::filtFilt): odd extension of 3 * (nfilt - 1) samples, steady-state initial conditions. */
export const filtfilt2 = ({ b, a }: { b: number[]; a: number[] }, x: number[]) => {
  const nf = 6;
  // lfilter_zi: solve (I - companion(a)^T) zi = b[1:] - a[1:] * b[0]
  const [m00, m01, m10, m11] = [1 + a[1], -1, a[2], 1];
  const [r0, r1] = [b[1] - a[1] * b[0], b[2] - a[2] * b[0]];
  const det = m00 * m11 - m01 * m10;
  const zi = [(r0 * m11 - m01 * r1) / det, (m00 * r1 - m10 * r0) / det];
  const ext = [
    ...Array.from({ length: nf }, (_, i) => 2 * x[0] - x[nf - i]),
    ...x,
    ...Array.from({ length: nf }, (_, i) => 2 * x.at(-1)! - x.at(-2 - i)!),
  ];
  const forward = lfilter2(b, a, ext, zi[0] * ext[0], zi[1] * ext[0]).toReversed();
  const backward = lfilter2(b, a, forward, zi[0] * forward[0], zi[1] * forward[0]).toReversed();
  return backward.slice(nf, -nf);
};

/**
 * ins::Stabilizer::GyroProcessImpl::inputGyroDataBatchWithBoxGravityFilter. Trapezoidal gyro
 * integration gives q_n (frame0 -> body_n); accel rotated into frame 0 is box-averaged over
 * GRAVITY_BOX_WINDOW_S (edge-clamped), normalised and Butterworth-smoothed, and each q_n is
 * levelled by an incrementally accumulated minimal rotation taking -g onto +z.
 * Returns undefined when the buffer is shorter than the box.
 */
export const boxGravityQuats = (ts: number[], acc: Vec3[], gyr: Vec3[]) => {
  const n = ts.length;
  const dt = (ts[n - 1] - ts[0]) / (n - 1);
  if (n < 3 || !(dt > 0) || GRAVITY_BOX_WINDOW_S / dt + 3 > n) {
    return;
  }
  let q: Quat = [0, 0, 0, 1];
  const quats: Quat[] = [q];
  const times = [ts[0]];
  const sums: Vec3[] = [[...acc[0]]];
  for (let j = 1; j < n; j++) {
    const step = ts[j] - ts[j - 1];
    const th = [0, 1, 2].map((k) => -(gyr[j - 1][k] + gyr[j][k]) * 0.5 * step) as Vec3;
    const angle = Math.hypot(...th);
    const s = angle <= 1e-7 ? 0.5 : Math.sin(angle * 0.5) / angle;
    const dq: Quat = [th[0] * s, th[1] * s, th[2] * s, angle <= 1e-7 ? 1 - (angle * angle) / 8 : Math.cos(angle * 0.5)];
    q = qnormalize(qmul(qnormalize(dq), q));
    quats.push(q);
    times.push((ts[j - 1] + ts[j]) * 0.5 + QUAT_TIME_OFFSET_S);
    const rotated = qrotate(qconj(q), acc[j]);
    const last = sums[j - 1];
    sums.push([last[0] + rotated[0], last[1] + rotated[1], last[2] + rotated[2]]);
  }

  const h = Math.floor((GRAVITY_BOX_WINDOW_S * 0.5) / dt);
  const gravity: Vec3[] = Array.from({ length: n });
  for (let c = h; c < n - h; c++) {
    gravity[c] = normalize([0, 1, 2].map((k) => sums[c + h][k] - sums[c - h][k]) as Vec3);
  }
  gravity.fill(gravity[h], 0, h);
  gravity.fill(gravity[n - h - 1], n - h);
  const filter = butter2Lowpass(GRAVITY_LOWPASS_HZ, 1 / dt);
  const smooth = [0, 1, 2].map((k) =>
    filtfilt2(
      filter,
      gravity.map((g) => g[k]),
    ),
  );

  let levelling: Quat = [0, 0, 0, 1];
  return {
    times,
    quats: quats.map((qn, i) => {
      const p = normalize(qrotate(levelling, [-smooth[0][i], -smooth[1][i], -smooth[2][i]]));
      const w = Math.sqrt((p[2] + 1) * 0.5);
      const step: Quat = w > 1e-6 ? [(p[1] * 0.5) / w, (-p[0] * 0.5) / w, 0, w] : [1, 0, 0, 0];
      levelling = qnormalize(qmul(step, levelling));
      return qmul(qn, qconj(levelling));
    }),
  };
};

// a zero gravity sample would otherwise stitch an all-black image without any error
const finite = (m: Mat3) => {
  if (m.flat().some((value) => !Number.isFinite(value))) {
    throw new Insta360UnsupportedError('IMU gives no usable gravity');
  }
  return m;
};

/**
 * Output-dir -> camera rotation, the way Studio orients a photo. ImageAssetSource::GetStabilizer
 * first tries the SequenceStabilizer: PrecomputeStabilization converts the IMU
 * (StabilizerUtil::CvtAcc/CvtGyro), drops saturated and non-increasing samples, rotates it by
 * the inverse lens-0 matrix before AlignAxes, runs the box-gravity filter, and the photo takes
 * the slerped quaternion at first_frame_timestamp. Buffers too short for the box fall back to
 * SingleGyroStabilizer: minimal rotation of the first sample's gravity, R_lens0 * AlignAxes(a).
 */
export const insta360Orientation = (trailer: Insta360Trailer): Mat3 => {
  const { imu, accRange, gyroRange, firstFrameUs, offset } = trailer;
  if (!imu) {
    // without gravity there is no "up", and an unlevelled X5 frame is not upright either
    throw new Insta360UnsupportedError('no IMU record in trailer');
  }
  const lens0 = rotateMatrix(offset[4], offset[5], offset[6]);
  const count = imu.ts.length;
  const acc: Vec3[] = [];
  const gyr: Vec3[] = [];
  for (let i = 0; i < count; i++) {
    const v = (k: number) => imu.raw[i * 6 + k] - 32_768;
    acc.push([0, 1, 2].map((k) => (accRange * v(k)) / 32_768) as Vec3);
    gyr.push([3, 4, 5].map((k) => (((gyroRange * v(k)) / 32_768) * Math.PI) / 180) as Vec3);
  }

  const sequenceFrame = mul(IMU_ALIGN, transpose(lens0));
  const gyroLimit = (gyroRange * Math.PI) / 180;
  const ts: number[] = [];
  const seqAcc: Vec3[] = [];
  const seqGyr: Vec3[] = [];
  let lastTs = -1;
  for (let i = 0; i < count; i++) {
    const keep =
      acc[i].every((value) => Math.abs(value) < accRange) &&
      gyr[i].every((value) => Math.abs(value) < gyroLimit) &&
      imu.ts[i] > lastTs;
    lastTs = Math.max(lastTs, imu.ts[i]);
    if (!keep) {
      continue;
    }

    ts.push(imu.ts[i] / 1e6);
    seqAcc.push(apply(sequenceFrame, acc[i]));
    seqGyr.push(apply(sequenceFrame, gyr[i]));
  }

  const sequence = firstFrameUs === undefined ? undefined : boxGravityQuats(ts, seqAcc, seqGyr);
  if (sequence) {
    const { times, quats } = sequence;
    const t = firstFrameUs! / 1e6;
    let k = times.findIndex((time) => time > t) - 1;
    k = Math.min(Math.max(k < -1 ? times.length - 2 : k, 0), times.length - 2);
    const f = Math.min(Math.max((t - times[k]) / (times[k + 1] - times[k]), 0), 1);
    const [q0, q1] = [quats[k], quats[k + 1]];
    const sign = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3] < 0 ? -1 : 1;
    const q = qnormalize(q0.map((value, i) => (1 - f) * value + f * sign * q1[i]) as Quat);
    return finite(mul(mul(SPHERE_IN, qmatrix(q)), SPHERE_OUT));
  }
  const levelled = rotationBetween(apply(mul(lens0, IMU_ALIGN), acc[0]), [0, 0, -1]);
  return finite(mul(mul(SPHERE_IN, transpose(levelled)), SPHERE_OUT));
};

const clamp = (value: number) => Math.min(Math.max(Math.round(value), 0), 255);

/** The trailer's NV12 preview as packed RGB (BT.601, limited range). */
export const insta360PreviewRgb = ({ width, height, nv12 }: NonNullable<Insta360Trailer['preview']>) => {
  const rgb = Buffer.alloc(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const luma = 1.164 * (nv12[y * width + x] - 16);
      const uv = width * height + (y >> 1) * width + (x & ~1);
      const u = nv12[uv] - 128;
      const v = nv12[uv + 1] - 128;
      const o = (y * width + x) * 3;
      rgb[o] = clamp(luma + 1.596 * v);
      rgb[o + 1] = clamp(luma - 0.392 * u - 0.813 * v);
      rgb[o + 2] = clamp(luma + 2.017 * u);
    }
  }
  return rgb;
};

// ---- projection -----------------------------------------------------------------------------

export type Insta360Lens = { rotation: Mat3; cx: number; cy: number; radius: number };

/** Both lenses for a lens crop of `lensSize` px, levelled by `level` (output dir -> camera). */
export const insta360Lenses = (trailer: Insta360Trailer, lensSize: number, level: Mat3) => {
  const [, r0, cx0, cy0, rx0, ry0, rz0, r1, cx1, cy1, rx1, ry1, rz1, calibWidth, , lensWord] = trailer.offset;
  const coeffs = COEFF_BY_LENSTYPE[lensWord & 0xff];
  if (!coeffs) {
    throw new Insta360UnsupportedError(`unsupported lens type ${lensWord & 0xff} (only the Insta360 X5 is supported)`);
  }
  const halfFov = HALF_FOV_BY_OFFSET_STATE[trailer.offsetState];
  if (halfFov === undefined) {
    throw new Insta360UnsupportedError(`unsupported lens guard or accessory (offset state ${trailer.offsetState})`);
  }
  if (!(calibWidth > 0 && r0 > 0 && r1 > 0)) {
    throw new Insta360UnsupportedError('invalid lens calibration');
  }
  const scale = (2 * lensSize) / calibWidth;
  const p = mul(axisRotation('x', 90), axisRotation('y', 90));
  const lenses: [Insta360Lens, Insta360Lens] = [
    { rotation: mul(mul(rotateMatrix(rx0, ry0, rz0), p), level), cx: cx0 * scale, cy: cy0 * scale, radius: r0 * scale },
    {
      rotation: mul(mul(mul(rotateMatrix(rx1, ry1, rz1), axisRotation('z', 180)), p), level),
      cx: cx1 * scale - lensSize, // lens 1's centre is given in the full side-by-side frame
      cy: cy1 * scale,
      radius: r1 * scale,
    },
  ];
  return { lenses, coeffs, halfFov, level };
};

/** Lens 0's share of a pixel: Studio's alpha where both lenses see it, the only lens that does otherwise. */
export const blendWeight = (valid0: boolean, valid1: boolean, alpha: number) => {
  const w0 = valid0 ? alpha : 0;
  const w1 = valid1 ? 1 - alpha : 0;
  return w0 + w1 > 0 ? w0 / (w0 + w1) : valid0 ? 1 : 0;
};

const mapPoly = (phi: number, c: number[]) => c[0] + phi * (c[1] + phi * (c[2] + phi * (c[3] + phi * c[4])));

/**
 * Per output pixel of a width x (width / 2) equirectangular image: where to sample each lens
 * (NaN when the ray is outside that lens) and lens 0's blend weight. Calls `visit` row by row
 * so callers never hold full-resolution float maps, and yields to the event loop every 64 rows
 * so a 40 s photo stitch doesn't starve job-lock renewal.
 */
export const forEachInsta360Row = async (
  width: number,
  height: number,
  lensData: ReturnType<typeof insta360Lenses>,
  visit: (y: number, u0: Float32Array, v0: Float32Array, u1: Float32Array, v1: Float32Array, w0: Float32Array) => void,
) => {
  const { lenses, coeffs, halfFov, level } = lensData;
  const mapMax = mapPoly(halfFov, coeffs);
  const toDeg = 180 / Math.PI;
  const [u0, v0, u1, v1, w0] = Array.from({ length: 5 }, () => new Float32Array(width));
  const us = [u0, u1];
  const vs = [v0, v1];
  for (let y = 0; y < height; y++) {
    const ph = (Math.PI * (y + 0.5)) / height;
    const sp = Math.sin(ph);
    const cp = Math.cos(ph);
    for (let x = 0; x < width; x++) {
      const th = (2 * Math.PI * (x + 0.5)) / width;
      const pos: Vec3 = [-sp * Math.sin(th), cp, sp * Math.cos(th)]; // Studio's vertex shader
      for (let l = 0; l < 2; l++) {
        const { rotation: m, cx, cy, radius } = lenses[l];
        const rx = m[0][0] * pos[0] + m[0][1] * pos[1] + m[0][2] * pos[2];
        const ry = m[1][0] * pos[0] + m[1][1] * pos[1] + m[1][2] * pos[2];
        const rz = m[2][0] * pos[0] + m[2][1] * pos[1] + m[2][2] * pos[2];
        const phi = Math.atan2(Math.hypot(rx, ry), rz) * toDeg;
        if (phi > halfFov) {
          us[l][x] = NaN;
          vs[l][x] = NaN;
          continue;
        }
        const r = (mapPoly(phi, coeffs) / mapMax) * radius;
        const theta = Math.atan2(ry, rx);
        us[l][x] = r * Math.cos(theta) + cx;
        vs[l][x] = r * Math.sin(theta) + cy;
      }
      // calAlpha on the levelled ray: 1 -> lens 0, 0 -> lens 1
      const [px, py, pz] = apply(level, pos);
      const blend =
        py < 0
          ? Math.min(Math.max((1 - py ** 4) * BLEND_ANGLE_DEG, NADIR_BLEND_FLOOR_DEG), BLEND_ANGLE_DEG)
          : BLEND_ANGLE_DEG;
      const phiB = Math.atan2(Math.hypot(px, py), -pz) * toDeg;
      const t = Math.min(Math.max((90 - phiB + blend / 2) / blend, 0), 1);
      w0[x] = t <= 0.5 ? 0.5 * (2 * t) ** ALPHA_LAMBDA : 1 - 0.5 * (2 * (1 - t)) ** ALPHA_LAMBDA;
    }
    visit(y, u0, v0, u1, v1, w0);
    if (y % 64 === 63) {
      await setImmediate();
    }
  }
};
