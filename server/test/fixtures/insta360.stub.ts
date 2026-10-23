/** Builders for synthetic Insta360 trailers (see src/utils/insta360.ts for the layout). */
const varint = (value: number) => {
  const bytes: number[] = [];
  do {
    bytes.push((value & 0x7f) | (value >= 0x80 ? 0x80 : 0));
    value = Math.floor(value / 128);
  } while (value > 0);
  return Buffer.from(bytes);
};

export const field = (number: number, value: number | Buffer) =>
  typeof value === 'number'
    ? Buffer.concat([varint(number * 8), varint(value)])
    : Buffer.concat([varint(number * 8 + 2), varint(value.length), value]);

export const record = (id: number, payload: Buffer) => {
  const header = Buffer.alloc(6);
  header.writeUInt16LE(id, 0);
  header.writeUInt32LE(payload.length, 2);
  return Buffer.concat([payload, header]);
};

/** An X5 calibration: two lenses, 10752x5376 calibration frame, lens type 113. */
export const INSTA360_OFFSET =
  '2_2645.947_2700.400_2678.170_0.717_0.092_89.898_2646.118_8071.690_2674.010_-1.108_0.204_90.231_10752_5376_1137';

/** Raw IMU samples (record 0x300 payload), `stepUs` apart, all reading gravity `[x, y, z]` in raw units off 32768. */
export const imuSamples = (
  count: number,
  stepUs: number,
  gravity: [number, number, number] = [0, 0, -1024],
  startUs = 1_000_000,
) => {
  const imu = Buffer.alloc(count * 20);
  for (let i = 0; i < count; i++) {
    imu.writeBigUInt64LE(BigInt(startUs + i * stepUs), i * 20);
    for (const [k, value] of [...gravity, 0, 0, 0].entries()) {
      imu.writeUInt16LE(32_768 + value, i * 20 + 8 + k * 2);
    }
  }
  return imu;
};

/** A trailer as parseInsta360Trailer receives it: the records, then the 78-byte tail. */
export const insta360Trailer = (...records: Buffer[]) => Buffer.concat([...records, Buffer.alloc(72)]);

/** The bytes a real file ends with: the trailer with its length and magic filled into the tail. */
export const insta360FileTail = (...records: Buffer[]) => {
  const trailer = insta360Trailer(...records);
  trailer.writeUInt32LE(trailer.length, trailer.length - 78 + 38);
  trailer.write('8db42d694ccc418790edff439fe026bf', trailer.length - 32, 'latin1');
  return trailer;
};
