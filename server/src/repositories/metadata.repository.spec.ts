import { describe, expect, it } from 'vitest';
import { ImmichTags, mergeInsta360TrailerTags } from 'src/repositories/metadata.repository.js';

describe('mergeInsta360TrailerTags', () => {
  const utc = { zone: 'UTC', tz: 'UTC', zoneSource: 'defaultVideosToUTC' } as ImmichTags;
  const warsaw = { GPSLatitude: 52.23, GPSLongitude: 21.01 } as ImmichTags;

  it('should infer the zone from trailer GPS instead of the UTC guess', () => {
    expect(mergeInsta360TrailerTags(utc, { ...warsaw, Model: 'Insta360 X5' })).toMatchObject({
      zone: 'Europe/Warsaw',
      zoneSource: 'GPSLatitude/GPSLongitude',
      Model: 'Insta360 X5',
    });
  });

  it('should keep a zone the base read actually found', () => {
    const tagged = { zone: 'UTC+2', zoneSource: 'OffsetTimeOriginal' } as ImmichTags;
    expect(mergeInsta360TrailerTags(tagged, warsaw)).toMatchObject({ zone: 'UTC+2' });
  });

  it('should keep the UTC guess without usable GPS', () => {
    expect(mergeInsta360TrailerTags(utc, {})).toMatchObject({ zone: 'UTC' });
    expect(mergeInsta360TrailerTags(utc, { GPSLatitude: 0, GPSLongitude: 0 } as ImmichTags)).toMatchObject({
      zone: 'UTC',
    });
    expect(mergeInsta360TrailerTags(utc, { GPSLatitude: 95, GPSLongitude: 21 } as ImmichTags)).toMatchObject({
      zone: 'UTC',
    });
  });
});
