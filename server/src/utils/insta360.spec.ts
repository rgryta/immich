import { describe, expect, it } from 'vitest';
import {
  Mat3,
  Vec3,
  blendWeight,
  boxGravityQuats,
  butter2Lowpass,
  filtfilt2,
  insta360Lenses,
  insta360Orientation,
  insta360PreviewRgb,
  insta360TrailerLength,
  parseInsta360Trailer,
  qrotate,
} from 'src/utils/insta360.js';

const degreesBetween = (a: Vec3, b: Vec3) =>
  (Math.acos(Math.min(1, Math.max(-1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * 180) / Math.PI;

// A camera turning at a steady rate: world-fixed gravity seen in the body frame rotates the other way.
const turningCamera = (n: number) => {
  const w: Vec3 = [0.3, -0.5, 0.8];
  const angle = Math.hypot(...w);
  const g0: Vec3 = [0.2, -0.3, -0.93];
  const norm = Math.hypot(...g0);
  const ts = Array.from({ length: n }, (_, i) => i * 0.001);
  const acc = ts.map((t) => {
    const s = Math.sin((angle * t) / 2) / angle;
    return qrotate(
      [-w[0] * s, -w[1] * s, -w[2] * s, Math.cos((angle * t) / 2)],
      [g0[0] / norm, g0[1] / norm, g0[2] / norm],
    );
  });
  return { ts, acc, gyr: ts.map(() => w) };
};

const varint = (value: number) => {
  const bytes: number[] = [];
  do {
    bytes.push((value & 0x7f) | (value >= 0x80 ? 0x80 : 0));
    value = Math.floor(value / 128);
  } while (value > 0);
  return Buffer.from(bytes);
};
const field = (number: number, value: number | Buffer) =>
  typeof value === 'number'
    ? Buffer.concat([varint(number * 8), varint(value)])
    : Buffer.concat([varint(number * 8 + 2), varint(value.length), value]);
const record = (id: number, payload: Buffer) => {
  const header = Buffer.alloc(6);
  header.writeUInt16LE(id, 0);
  header.writeUInt32LE(payload.length, 2);
  return Buffer.concat([payload, header]);
};
const OFFSET =
  '2_2645.947_2700.400_2678.170_0.717_0.092_89.898_2646.118_8071.690_2674.010_-1.108_0.204_90.231_10752_5376_1137';
const IDENTITY: Mat3 = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];
const lensFov = (offsetState: number) => {
  const metadata = Buffer.concat([field(5, Buffer.from(OFFSET)), field(68, offsetState)]);
  const trailer = parseInsta360Trailer(Buffer.concat([record(0x1_01, metadata), Buffer.alloc(72)]));
  return insta360Lenses(trailer, 1000, IDENTITY).halfFov;
};

describe('insta360', () => {
  it('filtfilt keeps a constant and matches MATLAB/scipy filtfilt', () => {
    const filter = butter2Lowpass(0.02, 1000);
    const constant = filtfilt2(
      filter,
      Array.from({ length: 5000 }, () => 0.3),
    );
    expect(Math.max(...constant.map((value) => Math.abs(value - 0.3)))).toBeLessThan(1e-6);

    // reference: scipy.signal.filtfilt(..., padtype='odd', padlen=6) on linspace(-1, 1, 5000)
    const ramp = filtfilt2(
      filter,
      Array.from({ length: 5000 }, (_, i) => -1 + (2 * i) / 4999),
    );
    for (const [i, expected] of [
      [0, -0.906286],
      [2500, -0.898521],
      [4999, -0.896801],
    ]) {
      expect(ramp[i]).toBeCloseTo(expected, 5);
    }
  });

  it('levels a turning camera to its measured gravity', () => {
    const { ts, acc, gyr } = turningCamera(12_000);
    const { quats } = boxGravityQuats(ts, acc, gyr)!;
    for (const k of [0, 4000, 11_999]) {
      const up = qrotate(quats[k], [0, 0, 1]);
      expect(degreesBetween(up, [-acc[k][0], -acc[k][1], -acc[k][2]])).toBeLessThan(0.05);
    }
  });

  it('falls back when the IMU buffer is shorter than the gravity box', () => {
    const { ts, acc, gyr } = turningCamera(5000);
    expect(boxGravityQuats(ts, acc, gyr)).toBeUndefined();
  });

  it('parses calibration, ranges and IMU from the trailer', () => {
    const metadata = Buffer.concat([
      field(5, Buffer.from(OFFSET)),
      field(24, 1_000_000),
      field(65, Buffer.concat([field(1, 32), field(2, 2000)])),
    ]);
    const imu = Buffer.alloc(40);
    for (let i = 0; i < 2; i++) {
      imu.writeBigUInt64LE(BigInt(1_000_000 + i * 1000), i * 20);
      for (const [k, value] of [32_768, 32_768, 32_768 - 1024, 32_768, 32_768, 32_768].entries())
        imu.writeUInt16LE(value, i * 20 + 8 + k * 2);
    }
    const trailer = parseInsta360Trailer(
      Buffer.concat([record(0x3_00, imu), record(0x1_01, metadata), Buffer.alloc(72)]),
    );
    expect(trailer.offset).toHaveLength(16);
    expect(trailer.offset[15]).toBe(1137);
    expect(trailer).toMatchObject({ firstFrameUs: 1_000_000, accRange: 32, gyroRange: 2000 });
    expect(trailer.imu?.ts).toEqual(new Float64Array([1_000_000, 1_001_000]));

    // two samples are far too few for the gravity box: single-gyro fallback, a proper rotation
    const rotation = insta360Orientation(trailer);
    const det =
      rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1]) -
      rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0]) +
      rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]);
    expect(det).toBeCloseTo(1, 9);
    expect(() => insta360Orientation({ ...trailer, imu: undefined })).toThrow('no IMU');
  });

  it('finds .insv records through the trailer directory table', () => {
    const metadata = field(5, Buffer.from(OFFSET));
    const imu = Buffer.alloc(20);
    const body = Buffer.concat([record(0x1_01, metadata), record(0x3_00, imu)]);
    const directory = Buffer.alloc(20);
    directory.writeUInt16LE(0x1, 0);
    directory.writeUInt32LE(metadata.length, 2);
    directory.writeUInt32LE(0, 6);
    directory.writeUInt16LE(0x3, 10);
    directory.writeUInt32LE(imu.length, 12);
    directory.writeUInt32LE(metadata.length + 6, 16);
    const trailer = parseInsta360Trailer(Buffer.concat([body, record(0x0, directory), Buffer.alloc(72)]));
    expect(trailer.offset).toHaveLength(16);
    expect(trailer.imu?.ts).toHaveLength(1);
  });

  it('reads the NV12 preview and converts it to RGB', () => {
    const header = Buffer.alloc(40);
    for (const [i, value] of [1, 40 + 6, 1, 0, 2, 2].entries()) header.writeUInt32LE(value, i * 4);
    // 2x2: black, white, mid grey, mid grey; one neutral chroma pair
    const preview = Buffer.concat([header, Buffer.from([16, 235, 126, 126, 128, 128])]);
    const trailer = parseInsta360Trailer(
      Buffer.concat([record(0x2_00, preview), record(0x1_01, field(5, Buffer.from(OFFSET))), Buffer.alloc(72)]),
    );
    expect(trailer.preview).toMatchObject({ width: 2, height: 2 });
    expect([...insta360PreviewRgb(trailer.preview!)]).toEqual([0, 0, 0, 255, 255, 255, 128, 128, 128, 128, 128, 128]);
  });

  it('trusts the trailer length only behind the Insta360 magic', () => {
    const tail = Buffer.alloc(78);
    tail.writeUInt32LE(150, 38); // a plausible length, but no magic
    expect(() => insta360TrailerLength(tail, 200)).toThrow('no Insta360 trailer');
    tail.write('8db42d694ccc418790edff439fe026bf', 46, 'latin1');
    expect(insta360TrailerLength(tail, 200)).toBe(150);
    expect(() => insta360TrailerLength(tail, 100)).toThrow('no Insta360 trailer');
  });

  it('refuses to level with a zero gravity sample instead of stitching black', () => {
    const imu = Buffer.alloc(20);
    imu.writeBigUInt64LE(1_000_000n, 0);
    for (const [k, value] of [32_768, 32_768, 32_768, 32_768, 32_768, 32_768].entries())
      imu.writeUInt16LE(value, 8 + k * 2);
    const trailer = parseInsta360Trailer(
      Buffer.concat([record(0x3_00, imu), record(0x1_01, field(5, Buffer.from(OFFSET))), Buffer.alloc(72)]),
    );
    expect(() => insta360Orientation(trailer)).toThrow('no usable gravity');
  });

  it('picks the lens FOV from the lens guard and refuses unmeasured accessories', () => {
    expect(lensFov(0)).toBe(100);
    expect(lensFov(5)).toBe(97.5);
    expect(() => lensFov(4)).toThrow('offset state 4');
  });

  it('gives each pixel to the lens that can see it', () => {
    expect(blendWeight(true, true, 0.25)).toBe(0.25);
    expect(blendWeight(true, false, 0)).toBe(1);
    expect(blendWeight(false, true, 1)).toBe(0);
    expect(blendWeight(false, false, 0.5)).toBe(0);
  });
});
