// SPDX-License-Identifier: AGPL-3.0-only
// Unicode 16.0 General_Category tables for ztok.
// Generated from DerivedGeneralCategory.txt (Unicode 16.0.0).
// DOD: sorted [lo,hi] ranges + binary search; no hash maps, no per-cp arrays.
// Each lookup is O(log R) where R ~= hundreds; ~10 compares worst case.
// PERF: An ASCII fast-path (cp < 128) check before binary search would cut
// the common case to a single compare; left as a Wave-C optimization, the
// integrator may add it inline in cl100k.zig where the call-site hottness
// is known.

const std = @import("std");

pub const Range = struct { lo: u21, hi: u21 };

const cat_Lu = [_]Range{
    .{ .lo = 0x0041, .hi = 0x005A },   .{ .lo = 0x00C0, .hi = 0x00D6 },   .{ .lo = 0x00D8, .hi = 0x00DE },   .{ .lo = 0x0100, .hi = 0x0100 },
    .{ .lo = 0x0102, .hi = 0x0102 },   .{ .lo = 0x0104, .hi = 0x0104 },   .{ .lo = 0x0106, .hi = 0x0106 },   .{ .lo = 0x0108, .hi = 0x0108 },
    .{ .lo = 0x010A, .hi = 0x010A },   .{ .lo = 0x010C, .hi = 0x010C },   .{ .lo = 0x010E, .hi = 0x010E },   .{ .lo = 0x0110, .hi = 0x0110 },
    .{ .lo = 0x0112, .hi = 0x0112 },   .{ .lo = 0x0114, .hi = 0x0114 },   .{ .lo = 0x0116, .hi = 0x0116 },   .{ .lo = 0x0118, .hi = 0x0118 },
    .{ .lo = 0x011A, .hi = 0x011A },   .{ .lo = 0x011C, .hi = 0x011C },   .{ .lo = 0x011E, .hi = 0x011E },   .{ .lo = 0x0120, .hi = 0x0120 },
    .{ .lo = 0x0122, .hi = 0x0122 },   .{ .lo = 0x0124, .hi = 0x0124 },   .{ .lo = 0x0126, .hi = 0x0126 },   .{ .lo = 0x0128, .hi = 0x0128 },
    .{ .lo = 0x012A, .hi = 0x012A },   .{ .lo = 0x012C, .hi = 0x012C },   .{ .lo = 0x012E, .hi = 0x012E },   .{ .lo = 0x0130, .hi = 0x0130 },
    .{ .lo = 0x0132, .hi = 0x0132 },   .{ .lo = 0x0134, .hi = 0x0134 },   .{ .lo = 0x0136, .hi = 0x0136 },   .{ .lo = 0x0139, .hi = 0x0139 },
    .{ .lo = 0x013B, .hi = 0x013B },   .{ .lo = 0x013D, .hi = 0x013D },   .{ .lo = 0x013F, .hi = 0x013F },   .{ .lo = 0x0141, .hi = 0x0141 },
    .{ .lo = 0x0143, .hi = 0x0143 },   .{ .lo = 0x0145, .hi = 0x0145 },   .{ .lo = 0x0147, .hi = 0x0147 },   .{ .lo = 0x014A, .hi = 0x014A },
    .{ .lo = 0x014C, .hi = 0x014C },   .{ .lo = 0x014E, .hi = 0x014E },   .{ .lo = 0x0150, .hi = 0x0150 },   .{ .lo = 0x0152, .hi = 0x0152 },
    .{ .lo = 0x0154, .hi = 0x0154 },   .{ .lo = 0x0156, .hi = 0x0156 },   .{ .lo = 0x0158, .hi = 0x0158 },   .{ .lo = 0x015A, .hi = 0x015A },
    .{ .lo = 0x015C, .hi = 0x015C },   .{ .lo = 0x015E, .hi = 0x015E },   .{ .lo = 0x0160, .hi = 0x0160 },   .{ .lo = 0x0162, .hi = 0x0162 },
    .{ .lo = 0x0164, .hi = 0x0164 },   .{ .lo = 0x0166, .hi = 0x0166 },   .{ .lo = 0x0168, .hi = 0x0168 },   .{ .lo = 0x016A, .hi = 0x016A },
    .{ .lo = 0x016C, .hi = 0x016C },   .{ .lo = 0x016E, .hi = 0x016E },   .{ .lo = 0x0170, .hi = 0x0170 },   .{ .lo = 0x0172, .hi = 0x0172 },
    .{ .lo = 0x0174, .hi = 0x0174 },   .{ .lo = 0x0176, .hi = 0x0176 },   .{ .lo = 0x0178, .hi = 0x0179 },   .{ .lo = 0x017B, .hi = 0x017B },
    .{ .lo = 0x017D, .hi = 0x017D },   .{ .lo = 0x0181, .hi = 0x0182 },   .{ .lo = 0x0184, .hi = 0x0184 },   .{ .lo = 0x0186, .hi = 0x0187 },
    .{ .lo = 0x0189, .hi = 0x018B },   .{ .lo = 0x018E, .hi = 0x0191 },   .{ .lo = 0x0193, .hi = 0x0194 },   .{ .lo = 0x0196, .hi = 0x0198 },
    .{ .lo = 0x019C, .hi = 0x019D },   .{ .lo = 0x019F, .hi = 0x01A0 },   .{ .lo = 0x01A2, .hi = 0x01A2 },   .{ .lo = 0x01A4, .hi = 0x01A4 },
    .{ .lo = 0x01A6, .hi = 0x01A7 },   .{ .lo = 0x01A9, .hi = 0x01A9 },   .{ .lo = 0x01AC, .hi = 0x01AC },   .{ .lo = 0x01AE, .hi = 0x01AF },
    .{ .lo = 0x01B1, .hi = 0x01B3 },   .{ .lo = 0x01B5, .hi = 0x01B5 },   .{ .lo = 0x01B7, .hi = 0x01B8 },   .{ .lo = 0x01BC, .hi = 0x01BC },
    .{ .lo = 0x01C4, .hi = 0x01C4 },   .{ .lo = 0x01C7, .hi = 0x01C7 },   .{ .lo = 0x01CA, .hi = 0x01CA },   .{ .lo = 0x01CD, .hi = 0x01CD },
    .{ .lo = 0x01CF, .hi = 0x01CF },   .{ .lo = 0x01D1, .hi = 0x01D1 },   .{ .lo = 0x01D3, .hi = 0x01D3 },   .{ .lo = 0x01D5, .hi = 0x01D5 },
    .{ .lo = 0x01D7, .hi = 0x01D7 },   .{ .lo = 0x01D9, .hi = 0x01D9 },   .{ .lo = 0x01DB, .hi = 0x01DB },   .{ .lo = 0x01DE, .hi = 0x01DE },
    .{ .lo = 0x01E0, .hi = 0x01E0 },   .{ .lo = 0x01E2, .hi = 0x01E2 },   .{ .lo = 0x01E4, .hi = 0x01E4 },   .{ .lo = 0x01E6, .hi = 0x01E6 },
    .{ .lo = 0x01E8, .hi = 0x01E8 },   .{ .lo = 0x01EA, .hi = 0x01EA },   .{ .lo = 0x01EC, .hi = 0x01EC },   .{ .lo = 0x01EE, .hi = 0x01EE },
    .{ .lo = 0x01F1, .hi = 0x01F1 },   .{ .lo = 0x01F4, .hi = 0x01F4 },   .{ .lo = 0x01F6, .hi = 0x01F8 },   .{ .lo = 0x01FA, .hi = 0x01FA },
    .{ .lo = 0x01FC, .hi = 0x01FC },   .{ .lo = 0x01FE, .hi = 0x01FE },   .{ .lo = 0x0200, .hi = 0x0200 },   .{ .lo = 0x0202, .hi = 0x0202 },
    .{ .lo = 0x0204, .hi = 0x0204 },   .{ .lo = 0x0206, .hi = 0x0206 },   .{ .lo = 0x0208, .hi = 0x0208 },   .{ .lo = 0x020A, .hi = 0x020A },
    .{ .lo = 0x020C, .hi = 0x020C },   .{ .lo = 0x020E, .hi = 0x020E },   .{ .lo = 0x0210, .hi = 0x0210 },   .{ .lo = 0x0212, .hi = 0x0212 },
    .{ .lo = 0x0214, .hi = 0x0214 },   .{ .lo = 0x0216, .hi = 0x0216 },   .{ .lo = 0x0218, .hi = 0x0218 },   .{ .lo = 0x021A, .hi = 0x021A },
    .{ .lo = 0x021C, .hi = 0x021C },   .{ .lo = 0x021E, .hi = 0x021E },   .{ .lo = 0x0220, .hi = 0x0220 },   .{ .lo = 0x0222, .hi = 0x0222 },
    .{ .lo = 0x0224, .hi = 0x0224 },   .{ .lo = 0x0226, .hi = 0x0226 },   .{ .lo = 0x0228, .hi = 0x0228 },   .{ .lo = 0x022A, .hi = 0x022A },
    .{ .lo = 0x022C, .hi = 0x022C },   .{ .lo = 0x022E, .hi = 0x022E },   .{ .lo = 0x0230, .hi = 0x0230 },   .{ .lo = 0x0232, .hi = 0x0232 },
    .{ .lo = 0x023A, .hi = 0x023B },   .{ .lo = 0x023D, .hi = 0x023E },   .{ .lo = 0x0241, .hi = 0x0241 },   .{ .lo = 0x0243, .hi = 0x0246 },
    .{ .lo = 0x0248, .hi = 0x0248 },   .{ .lo = 0x024A, .hi = 0x024A },   .{ .lo = 0x024C, .hi = 0x024C },   .{ .lo = 0x024E, .hi = 0x024E },
    .{ .lo = 0x0370, .hi = 0x0370 },   .{ .lo = 0x0372, .hi = 0x0372 },   .{ .lo = 0x0376, .hi = 0x0376 },   .{ .lo = 0x037F, .hi = 0x037F },
    .{ .lo = 0x0386, .hi = 0x0386 },   .{ .lo = 0x0388, .hi = 0x038A },   .{ .lo = 0x038C, .hi = 0x038C },   .{ .lo = 0x038E, .hi = 0x038F },
    .{ .lo = 0x0391, .hi = 0x03A1 },   .{ .lo = 0x03A3, .hi = 0x03AB },   .{ .lo = 0x03CF, .hi = 0x03CF },   .{ .lo = 0x03D2, .hi = 0x03D4 },
    .{ .lo = 0x03D8, .hi = 0x03D8 },   .{ .lo = 0x03DA, .hi = 0x03DA },   .{ .lo = 0x03DC, .hi = 0x03DC },   .{ .lo = 0x03DE, .hi = 0x03DE },
    .{ .lo = 0x03E0, .hi = 0x03E0 },   .{ .lo = 0x03E2, .hi = 0x03E2 },   .{ .lo = 0x03E4, .hi = 0x03E4 },   .{ .lo = 0x03E6, .hi = 0x03E6 },
    .{ .lo = 0x03E8, .hi = 0x03E8 },   .{ .lo = 0x03EA, .hi = 0x03EA },   .{ .lo = 0x03EC, .hi = 0x03EC },   .{ .lo = 0x03EE, .hi = 0x03EE },
    .{ .lo = 0x03F4, .hi = 0x03F4 },   .{ .lo = 0x03F7, .hi = 0x03F7 },   .{ .lo = 0x03F9, .hi = 0x03FA },   .{ .lo = 0x03FD, .hi = 0x042F },
    .{ .lo = 0x0460, .hi = 0x0460 },   .{ .lo = 0x0462, .hi = 0x0462 },   .{ .lo = 0x0464, .hi = 0x0464 },   .{ .lo = 0x0466, .hi = 0x0466 },
    .{ .lo = 0x0468, .hi = 0x0468 },   .{ .lo = 0x046A, .hi = 0x046A },   .{ .lo = 0x046C, .hi = 0x046C },   .{ .lo = 0x046E, .hi = 0x046E },
    .{ .lo = 0x0470, .hi = 0x0470 },   .{ .lo = 0x0472, .hi = 0x0472 },   .{ .lo = 0x0474, .hi = 0x0474 },   .{ .lo = 0x0476, .hi = 0x0476 },
    .{ .lo = 0x0478, .hi = 0x0478 },   .{ .lo = 0x047A, .hi = 0x047A },   .{ .lo = 0x047C, .hi = 0x047C },   .{ .lo = 0x047E, .hi = 0x047E },
    .{ .lo = 0x0480, .hi = 0x0480 },   .{ .lo = 0x048A, .hi = 0x048A },   .{ .lo = 0x048C, .hi = 0x048C },   .{ .lo = 0x048E, .hi = 0x048E },
    .{ .lo = 0x0490, .hi = 0x0490 },   .{ .lo = 0x0492, .hi = 0x0492 },   .{ .lo = 0x0494, .hi = 0x0494 },   .{ .lo = 0x0496, .hi = 0x0496 },
    .{ .lo = 0x0498, .hi = 0x0498 },   .{ .lo = 0x049A, .hi = 0x049A },   .{ .lo = 0x049C, .hi = 0x049C },   .{ .lo = 0x049E, .hi = 0x049E },
    .{ .lo = 0x04A0, .hi = 0x04A0 },   .{ .lo = 0x04A2, .hi = 0x04A2 },   .{ .lo = 0x04A4, .hi = 0x04A4 },   .{ .lo = 0x04A6, .hi = 0x04A6 },
    .{ .lo = 0x04A8, .hi = 0x04A8 },   .{ .lo = 0x04AA, .hi = 0x04AA },   .{ .lo = 0x04AC, .hi = 0x04AC },   .{ .lo = 0x04AE, .hi = 0x04AE },
    .{ .lo = 0x04B0, .hi = 0x04B0 },   .{ .lo = 0x04B2, .hi = 0x04B2 },   .{ .lo = 0x04B4, .hi = 0x04B4 },   .{ .lo = 0x04B6, .hi = 0x04B6 },
    .{ .lo = 0x04B8, .hi = 0x04B8 },   .{ .lo = 0x04BA, .hi = 0x04BA },   .{ .lo = 0x04BC, .hi = 0x04BC },   .{ .lo = 0x04BE, .hi = 0x04BE },
    .{ .lo = 0x04C0, .hi = 0x04C1 },   .{ .lo = 0x04C3, .hi = 0x04C3 },   .{ .lo = 0x04C5, .hi = 0x04C5 },   .{ .lo = 0x04C7, .hi = 0x04C7 },
    .{ .lo = 0x04C9, .hi = 0x04C9 },   .{ .lo = 0x04CB, .hi = 0x04CB },   .{ .lo = 0x04CD, .hi = 0x04CD },   .{ .lo = 0x04D0, .hi = 0x04D0 },
    .{ .lo = 0x04D2, .hi = 0x04D2 },   .{ .lo = 0x04D4, .hi = 0x04D4 },   .{ .lo = 0x04D6, .hi = 0x04D6 },   .{ .lo = 0x04D8, .hi = 0x04D8 },
    .{ .lo = 0x04DA, .hi = 0x04DA },   .{ .lo = 0x04DC, .hi = 0x04DC },   .{ .lo = 0x04DE, .hi = 0x04DE },   .{ .lo = 0x04E0, .hi = 0x04E0 },
    .{ .lo = 0x04E2, .hi = 0x04E2 },   .{ .lo = 0x04E4, .hi = 0x04E4 },   .{ .lo = 0x04E6, .hi = 0x04E6 },   .{ .lo = 0x04E8, .hi = 0x04E8 },
    .{ .lo = 0x04EA, .hi = 0x04EA },   .{ .lo = 0x04EC, .hi = 0x04EC },   .{ .lo = 0x04EE, .hi = 0x04EE },   .{ .lo = 0x04F0, .hi = 0x04F0 },
    .{ .lo = 0x04F2, .hi = 0x04F2 },   .{ .lo = 0x04F4, .hi = 0x04F4 },   .{ .lo = 0x04F6, .hi = 0x04F6 },   .{ .lo = 0x04F8, .hi = 0x04F8 },
    .{ .lo = 0x04FA, .hi = 0x04FA },   .{ .lo = 0x04FC, .hi = 0x04FC },   .{ .lo = 0x04FE, .hi = 0x04FE },   .{ .lo = 0x0500, .hi = 0x0500 },
    .{ .lo = 0x0502, .hi = 0x0502 },   .{ .lo = 0x0504, .hi = 0x0504 },   .{ .lo = 0x0506, .hi = 0x0506 },   .{ .lo = 0x0508, .hi = 0x0508 },
    .{ .lo = 0x050A, .hi = 0x050A },   .{ .lo = 0x050C, .hi = 0x050C },   .{ .lo = 0x050E, .hi = 0x050E },   .{ .lo = 0x0510, .hi = 0x0510 },
    .{ .lo = 0x0512, .hi = 0x0512 },   .{ .lo = 0x0514, .hi = 0x0514 },   .{ .lo = 0x0516, .hi = 0x0516 },   .{ .lo = 0x0518, .hi = 0x0518 },
    .{ .lo = 0x051A, .hi = 0x051A },   .{ .lo = 0x051C, .hi = 0x051C },   .{ .lo = 0x051E, .hi = 0x051E },   .{ .lo = 0x0520, .hi = 0x0520 },
    .{ .lo = 0x0522, .hi = 0x0522 },   .{ .lo = 0x0524, .hi = 0x0524 },   .{ .lo = 0x0526, .hi = 0x0526 },   .{ .lo = 0x0528, .hi = 0x0528 },
    .{ .lo = 0x052A, .hi = 0x052A },   .{ .lo = 0x052C, .hi = 0x052C },   .{ .lo = 0x052E, .hi = 0x052E },   .{ .lo = 0x0531, .hi = 0x0556 },
    .{ .lo = 0x10A0, .hi = 0x10C5 },   .{ .lo = 0x10C7, .hi = 0x10C7 },   .{ .lo = 0x10CD, .hi = 0x10CD },   .{ .lo = 0x13A0, .hi = 0x13F5 },
    .{ .lo = 0x1C89, .hi = 0x1C89 },   .{ .lo = 0x1C90, .hi = 0x1CBA },   .{ .lo = 0x1CBD, .hi = 0x1CBF },   .{ .lo = 0x1E00, .hi = 0x1E00 },
    .{ .lo = 0x1E02, .hi = 0x1E02 },   .{ .lo = 0x1E04, .hi = 0x1E04 },   .{ .lo = 0x1E06, .hi = 0x1E06 },   .{ .lo = 0x1E08, .hi = 0x1E08 },
    .{ .lo = 0x1E0A, .hi = 0x1E0A },   .{ .lo = 0x1E0C, .hi = 0x1E0C },   .{ .lo = 0x1E0E, .hi = 0x1E0E },   .{ .lo = 0x1E10, .hi = 0x1E10 },
    .{ .lo = 0x1E12, .hi = 0x1E12 },   .{ .lo = 0x1E14, .hi = 0x1E14 },   .{ .lo = 0x1E16, .hi = 0x1E16 },   .{ .lo = 0x1E18, .hi = 0x1E18 },
    .{ .lo = 0x1E1A, .hi = 0x1E1A },   .{ .lo = 0x1E1C, .hi = 0x1E1C },   .{ .lo = 0x1E1E, .hi = 0x1E1E },   .{ .lo = 0x1E20, .hi = 0x1E20 },
    .{ .lo = 0x1E22, .hi = 0x1E22 },   .{ .lo = 0x1E24, .hi = 0x1E24 },   .{ .lo = 0x1E26, .hi = 0x1E26 },   .{ .lo = 0x1E28, .hi = 0x1E28 },
    .{ .lo = 0x1E2A, .hi = 0x1E2A },   .{ .lo = 0x1E2C, .hi = 0x1E2C },   .{ .lo = 0x1E2E, .hi = 0x1E2E },   .{ .lo = 0x1E30, .hi = 0x1E30 },
    .{ .lo = 0x1E32, .hi = 0x1E32 },   .{ .lo = 0x1E34, .hi = 0x1E34 },   .{ .lo = 0x1E36, .hi = 0x1E36 },   .{ .lo = 0x1E38, .hi = 0x1E38 },
    .{ .lo = 0x1E3A, .hi = 0x1E3A },   .{ .lo = 0x1E3C, .hi = 0x1E3C },   .{ .lo = 0x1E3E, .hi = 0x1E3E },   .{ .lo = 0x1E40, .hi = 0x1E40 },
    .{ .lo = 0x1E42, .hi = 0x1E42 },   .{ .lo = 0x1E44, .hi = 0x1E44 },   .{ .lo = 0x1E46, .hi = 0x1E46 },   .{ .lo = 0x1E48, .hi = 0x1E48 },
    .{ .lo = 0x1E4A, .hi = 0x1E4A },   .{ .lo = 0x1E4C, .hi = 0x1E4C },   .{ .lo = 0x1E4E, .hi = 0x1E4E },   .{ .lo = 0x1E50, .hi = 0x1E50 },
    .{ .lo = 0x1E52, .hi = 0x1E52 },   .{ .lo = 0x1E54, .hi = 0x1E54 },   .{ .lo = 0x1E56, .hi = 0x1E56 },   .{ .lo = 0x1E58, .hi = 0x1E58 },
    .{ .lo = 0x1E5A, .hi = 0x1E5A },   .{ .lo = 0x1E5C, .hi = 0x1E5C },   .{ .lo = 0x1E5E, .hi = 0x1E5E },   .{ .lo = 0x1E60, .hi = 0x1E60 },
    .{ .lo = 0x1E62, .hi = 0x1E62 },   .{ .lo = 0x1E64, .hi = 0x1E64 },   .{ .lo = 0x1E66, .hi = 0x1E66 },   .{ .lo = 0x1E68, .hi = 0x1E68 },
    .{ .lo = 0x1E6A, .hi = 0x1E6A },   .{ .lo = 0x1E6C, .hi = 0x1E6C },   .{ .lo = 0x1E6E, .hi = 0x1E6E },   .{ .lo = 0x1E70, .hi = 0x1E70 },
    .{ .lo = 0x1E72, .hi = 0x1E72 },   .{ .lo = 0x1E74, .hi = 0x1E74 },   .{ .lo = 0x1E76, .hi = 0x1E76 },   .{ .lo = 0x1E78, .hi = 0x1E78 },
    .{ .lo = 0x1E7A, .hi = 0x1E7A },   .{ .lo = 0x1E7C, .hi = 0x1E7C },   .{ .lo = 0x1E7E, .hi = 0x1E7E },   .{ .lo = 0x1E80, .hi = 0x1E80 },
    .{ .lo = 0x1E82, .hi = 0x1E82 },   .{ .lo = 0x1E84, .hi = 0x1E84 },   .{ .lo = 0x1E86, .hi = 0x1E86 },   .{ .lo = 0x1E88, .hi = 0x1E88 },
    .{ .lo = 0x1E8A, .hi = 0x1E8A },   .{ .lo = 0x1E8C, .hi = 0x1E8C },   .{ .lo = 0x1E8E, .hi = 0x1E8E },   .{ .lo = 0x1E90, .hi = 0x1E90 },
    .{ .lo = 0x1E92, .hi = 0x1E92 },   .{ .lo = 0x1E94, .hi = 0x1E94 },   .{ .lo = 0x1E9E, .hi = 0x1E9E },   .{ .lo = 0x1EA0, .hi = 0x1EA0 },
    .{ .lo = 0x1EA2, .hi = 0x1EA2 },   .{ .lo = 0x1EA4, .hi = 0x1EA4 },   .{ .lo = 0x1EA6, .hi = 0x1EA6 },   .{ .lo = 0x1EA8, .hi = 0x1EA8 },
    .{ .lo = 0x1EAA, .hi = 0x1EAA },   .{ .lo = 0x1EAC, .hi = 0x1EAC },   .{ .lo = 0x1EAE, .hi = 0x1EAE },   .{ .lo = 0x1EB0, .hi = 0x1EB0 },
    .{ .lo = 0x1EB2, .hi = 0x1EB2 },   .{ .lo = 0x1EB4, .hi = 0x1EB4 },   .{ .lo = 0x1EB6, .hi = 0x1EB6 },   .{ .lo = 0x1EB8, .hi = 0x1EB8 },
    .{ .lo = 0x1EBA, .hi = 0x1EBA },   .{ .lo = 0x1EBC, .hi = 0x1EBC },   .{ .lo = 0x1EBE, .hi = 0x1EBE },   .{ .lo = 0x1EC0, .hi = 0x1EC0 },
    .{ .lo = 0x1EC2, .hi = 0x1EC2 },   .{ .lo = 0x1EC4, .hi = 0x1EC4 },   .{ .lo = 0x1EC6, .hi = 0x1EC6 },   .{ .lo = 0x1EC8, .hi = 0x1EC8 },
    .{ .lo = 0x1ECA, .hi = 0x1ECA },   .{ .lo = 0x1ECC, .hi = 0x1ECC },   .{ .lo = 0x1ECE, .hi = 0x1ECE },   .{ .lo = 0x1ED0, .hi = 0x1ED0 },
    .{ .lo = 0x1ED2, .hi = 0x1ED2 },   .{ .lo = 0x1ED4, .hi = 0x1ED4 },   .{ .lo = 0x1ED6, .hi = 0x1ED6 },   .{ .lo = 0x1ED8, .hi = 0x1ED8 },
    .{ .lo = 0x1EDA, .hi = 0x1EDA },   .{ .lo = 0x1EDC, .hi = 0x1EDC },   .{ .lo = 0x1EDE, .hi = 0x1EDE },   .{ .lo = 0x1EE0, .hi = 0x1EE0 },
    .{ .lo = 0x1EE2, .hi = 0x1EE2 },   .{ .lo = 0x1EE4, .hi = 0x1EE4 },   .{ .lo = 0x1EE6, .hi = 0x1EE6 },   .{ .lo = 0x1EE8, .hi = 0x1EE8 },
    .{ .lo = 0x1EEA, .hi = 0x1EEA },   .{ .lo = 0x1EEC, .hi = 0x1EEC },   .{ .lo = 0x1EEE, .hi = 0x1EEE },   .{ .lo = 0x1EF0, .hi = 0x1EF0 },
    .{ .lo = 0x1EF2, .hi = 0x1EF2 },   .{ .lo = 0x1EF4, .hi = 0x1EF4 },   .{ .lo = 0x1EF6, .hi = 0x1EF6 },   .{ .lo = 0x1EF8, .hi = 0x1EF8 },
    .{ .lo = 0x1EFA, .hi = 0x1EFA },   .{ .lo = 0x1EFC, .hi = 0x1EFC },   .{ .lo = 0x1EFE, .hi = 0x1EFE },   .{ .lo = 0x1F08, .hi = 0x1F0F },
    .{ .lo = 0x1F18, .hi = 0x1F1D },   .{ .lo = 0x1F28, .hi = 0x1F2F },   .{ .lo = 0x1F38, .hi = 0x1F3F },   .{ .lo = 0x1F48, .hi = 0x1F4D },
    .{ .lo = 0x1F59, .hi = 0x1F59 },   .{ .lo = 0x1F5B, .hi = 0x1F5B },   .{ .lo = 0x1F5D, .hi = 0x1F5D },   .{ .lo = 0x1F5F, .hi = 0x1F5F },
    .{ .lo = 0x1F68, .hi = 0x1F6F },   .{ .lo = 0x1FB8, .hi = 0x1FBB },   .{ .lo = 0x1FC8, .hi = 0x1FCB },   .{ .lo = 0x1FD8, .hi = 0x1FDB },
    .{ .lo = 0x1FE8, .hi = 0x1FEC },   .{ .lo = 0x1FF8, .hi = 0x1FFB },   .{ .lo = 0x2102, .hi = 0x2102 },   .{ .lo = 0x2107, .hi = 0x2107 },
    .{ .lo = 0x210B, .hi = 0x210D },   .{ .lo = 0x2110, .hi = 0x2112 },   .{ .lo = 0x2115, .hi = 0x2115 },   .{ .lo = 0x2119, .hi = 0x211D },
    .{ .lo = 0x2124, .hi = 0x2124 },   .{ .lo = 0x2126, .hi = 0x2126 },   .{ .lo = 0x2128, .hi = 0x2128 },   .{ .lo = 0x212A, .hi = 0x212D },
    .{ .lo = 0x2130, .hi = 0x2133 },   .{ .lo = 0x213E, .hi = 0x213F },   .{ .lo = 0x2145, .hi = 0x2145 },   .{ .lo = 0x2183, .hi = 0x2183 },
    .{ .lo = 0x2C00, .hi = 0x2C2F },   .{ .lo = 0x2C60, .hi = 0x2C60 },   .{ .lo = 0x2C62, .hi = 0x2C64 },   .{ .lo = 0x2C67, .hi = 0x2C67 },
    .{ .lo = 0x2C69, .hi = 0x2C69 },   .{ .lo = 0x2C6B, .hi = 0x2C6B },   .{ .lo = 0x2C6D, .hi = 0x2C70 },   .{ .lo = 0x2C72, .hi = 0x2C72 },
    .{ .lo = 0x2C75, .hi = 0x2C75 },   .{ .lo = 0x2C7E, .hi = 0x2C80 },   .{ .lo = 0x2C82, .hi = 0x2C82 },   .{ .lo = 0x2C84, .hi = 0x2C84 },
    .{ .lo = 0x2C86, .hi = 0x2C86 },   .{ .lo = 0x2C88, .hi = 0x2C88 },   .{ .lo = 0x2C8A, .hi = 0x2C8A },   .{ .lo = 0x2C8C, .hi = 0x2C8C },
    .{ .lo = 0x2C8E, .hi = 0x2C8E },   .{ .lo = 0x2C90, .hi = 0x2C90 },   .{ .lo = 0x2C92, .hi = 0x2C92 },   .{ .lo = 0x2C94, .hi = 0x2C94 },
    .{ .lo = 0x2C96, .hi = 0x2C96 },   .{ .lo = 0x2C98, .hi = 0x2C98 },   .{ .lo = 0x2C9A, .hi = 0x2C9A },   .{ .lo = 0x2C9C, .hi = 0x2C9C },
    .{ .lo = 0x2C9E, .hi = 0x2C9E },   .{ .lo = 0x2CA0, .hi = 0x2CA0 },   .{ .lo = 0x2CA2, .hi = 0x2CA2 },   .{ .lo = 0x2CA4, .hi = 0x2CA4 },
    .{ .lo = 0x2CA6, .hi = 0x2CA6 },   .{ .lo = 0x2CA8, .hi = 0x2CA8 },   .{ .lo = 0x2CAA, .hi = 0x2CAA },   .{ .lo = 0x2CAC, .hi = 0x2CAC },
    .{ .lo = 0x2CAE, .hi = 0x2CAE },   .{ .lo = 0x2CB0, .hi = 0x2CB0 },   .{ .lo = 0x2CB2, .hi = 0x2CB2 },   .{ .lo = 0x2CB4, .hi = 0x2CB4 },
    .{ .lo = 0x2CB6, .hi = 0x2CB6 },   .{ .lo = 0x2CB8, .hi = 0x2CB8 },   .{ .lo = 0x2CBA, .hi = 0x2CBA },   .{ .lo = 0x2CBC, .hi = 0x2CBC },
    .{ .lo = 0x2CBE, .hi = 0x2CBE },   .{ .lo = 0x2CC0, .hi = 0x2CC0 },   .{ .lo = 0x2CC2, .hi = 0x2CC2 },   .{ .lo = 0x2CC4, .hi = 0x2CC4 },
    .{ .lo = 0x2CC6, .hi = 0x2CC6 },   .{ .lo = 0x2CC8, .hi = 0x2CC8 },   .{ .lo = 0x2CCA, .hi = 0x2CCA },   .{ .lo = 0x2CCC, .hi = 0x2CCC },
    .{ .lo = 0x2CCE, .hi = 0x2CCE },   .{ .lo = 0x2CD0, .hi = 0x2CD0 },   .{ .lo = 0x2CD2, .hi = 0x2CD2 },   .{ .lo = 0x2CD4, .hi = 0x2CD4 },
    .{ .lo = 0x2CD6, .hi = 0x2CD6 },   .{ .lo = 0x2CD8, .hi = 0x2CD8 },   .{ .lo = 0x2CDA, .hi = 0x2CDA },   .{ .lo = 0x2CDC, .hi = 0x2CDC },
    .{ .lo = 0x2CDE, .hi = 0x2CDE },   .{ .lo = 0x2CE0, .hi = 0x2CE0 },   .{ .lo = 0x2CE2, .hi = 0x2CE2 },   .{ .lo = 0x2CEB, .hi = 0x2CEB },
    .{ .lo = 0x2CED, .hi = 0x2CED },   .{ .lo = 0x2CF2, .hi = 0x2CF2 },   .{ .lo = 0xA640, .hi = 0xA640 },   .{ .lo = 0xA642, .hi = 0xA642 },
    .{ .lo = 0xA644, .hi = 0xA644 },   .{ .lo = 0xA646, .hi = 0xA646 },   .{ .lo = 0xA648, .hi = 0xA648 },   .{ .lo = 0xA64A, .hi = 0xA64A },
    .{ .lo = 0xA64C, .hi = 0xA64C },   .{ .lo = 0xA64E, .hi = 0xA64E },   .{ .lo = 0xA650, .hi = 0xA650 },   .{ .lo = 0xA652, .hi = 0xA652 },
    .{ .lo = 0xA654, .hi = 0xA654 },   .{ .lo = 0xA656, .hi = 0xA656 },   .{ .lo = 0xA658, .hi = 0xA658 },   .{ .lo = 0xA65A, .hi = 0xA65A },
    .{ .lo = 0xA65C, .hi = 0xA65C },   .{ .lo = 0xA65E, .hi = 0xA65E },   .{ .lo = 0xA660, .hi = 0xA660 },   .{ .lo = 0xA662, .hi = 0xA662 },
    .{ .lo = 0xA664, .hi = 0xA664 },   .{ .lo = 0xA666, .hi = 0xA666 },   .{ .lo = 0xA668, .hi = 0xA668 },   .{ .lo = 0xA66A, .hi = 0xA66A },
    .{ .lo = 0xA66C, .hi = 0xA66C },   .{ .lo = 0xA680, .hi = 0xA680 },   .{ .lo = 0xA682, .hi = 0xA682 },   .{ .lo = 0xA684, .hi = 0xA684 },
    .{ .lo = 0xA686, .hi = 0xA686 },   .{ .lo = 0xA688, .hi = 0xA688 },   .{ .lo = 0xA68A, .hi = 0xA68A },   .{ .lo = 0xA68C, .hi = 0xA68C },
    .{ .lo = 0xA68E, .hi = 0xA68E },   .{ .lo = 0xA690, .hi = 0xA690 },   .{ .lo = 0xA692, .hi = 0xA692 },   .{ .lo = 0xA694, .hi = 0xA694 },
    .{ .lo = 0xA696, .hi = 0xA696 },   .{ .lo = 0xA698, .hi = 0xA698 },   .{ .lo = 0xA69A, .hi = 0xA69A },   .{ .lo = 0xA722, .hi = 0xA722 },
    .{ .lo = 0xA724, .hi = 0xA724 },   .{ .lo = 0xA726, .hi = 0xA726 },   .{ .lo = 0xA728, .hi = 0xA728 },   .{ .lo = 0xA72A, .hi = 0xA72A },
    .{ .lo = 0xA72C, .hi = 0xA72C },   .{ .lo = 0xA72E, .hi = 0xA72E },   .{ .lo = 0xA732, .hi = 0xA732 },   .{ .lo = 0xA734, .hi = 0xA734 },
    .{ .lo = 0xA736, .hi = 0xA736 },   .{ .lo = 0xA738, .hi = 0xA738 },   .{ .lo = 0xA73A, .hi = 0xA73A },   .{ .lo = 0xA73C, .hi = 0xA73C },
    .{ .lo = 0xA73E, .hi = 0xA73E },   .{ .lo = 0xA740, .hi = 0xA740 },   .{ .lo = 0xA742, .hi = 0xA742 },   .{ .lo = 0xA744, .hi = 0xA744 },
    .{ .lo = 0xA746, .hi = 0xA746 },   .{ .lo = 0xA748, .hi = 0xA748 },   .{ .lo = 0xA74A, .hi = 0xA74A },   .{ .lo = 0xA74C, .hi = 0xA74C },
    .{ .lo = 0xA74E, .hi = 0xA74E },   .{ .lo = 0xA750, .hi = 0xA750 },   .{ .lo = 0xA752, .hi = 0xA752 },   .{ .lo = 0xA754, .hi = 0xA754 },
    .{ .lo = 0xA756, .hi = 0xA756 },   .{ .lo = 0xA758, .hi = 0xA758 },   .{ .lo = 0xA75A, .hi = 0xA75A },   .{ .lo = 0xA75C, .hi = 0xA75C },
    .{ .lo = 0xA75E, .hi = 0xA75E },   .{ .lo = 0xA760, .hi = 0xA760 },   .{ .lo = 0xA762, .hi = 0xA762 },   .{ .lo = 0xA764, .hi = 0xA764 },
    .{ .lo = 0xA766, .hi = 0xA766 },   .{ .lo = 0xA768, .hi = 0xA768 },   .{ .lo = 0xA76A, .hi = 0xA76A },   .{ .lo = 0xA76C, .hi = 0xA76C },
    .{ .lo = 0xA76E, .hi = 0xA76E },   .{ .lo = 0xA779, .hi = 0xA779 },   .{ .lo = 0xA77B, .hi = 0xA77B },   .{ .lo = 0xA77D, .hi = 0xA77E },
    .{ .lo = 0xA780, .hi = 0xA780 },   .{ .lo = 0xA782, .hi = 0xA782 },   .{ .lo = 0xA784, .hi = 0xA784 },   .{ .lo = 0xA786, .hi = 0xA786 },
    .{ .lo = 0xA78B, .hi = 0xA78B },   .{ .lo = 0xA78D, .hi = 0xA78D },   .{ .lo = 0xA790, .hi = 0xA790 },   .{ .lo = 0xA792, .hi = 0xA792 },
    .{ .lo = 0xA796, .hi = 0xA796 },   .{ .lo = 0xA798, .hi = 0xA798 },   .{ .lo = 0xA79A, .hi = 0xA79A },   .{ .lo = 0xA79C, .hi = 0xA79C },
    .{ .lo = 0xA79E, .hi = 0xA79E },   .{ .lo = 0xA7A0, .hi = 0xA7A0 },   .{ .lo = 0xA7A2, .hi = 0xA7A2 },   .{ .lo = 0xA7A4, .hi = 0xA7A4 },
    .{ .lo = 0xA7A6, .hi = 0xA7A6 },   .{ .lo = 0xA7A8, .hi = 0xA7A8 },   .{ .lo = 0xA7AA, .hi = 0xA7AE },   .{ .lo = 0xA7B0, .hi = 0xA7B4 },
    .{ .lo = 0xA7B6, .hi = 0xA7B6 },   .{ .lo = 0xA7B8, .hi = 0xA7B8 },   .{ .lo = 0xA7BA, .hi = 0xA7BA },   .{ .lo = 0xA7BC, .hi = 0xA7BC },
    .{ .lo = 0xA7BE, .hi = 0xA7BE },   .{ .lo = 0xA7C0, .hi = 0xA7C0 },   .{ .lo = 0xA7C2, .hi = 0xA7C2 },   .{ .lo = 0xA7C4, .hi = 0xA7C7 },
    .{ .lo = 0xA7C9, .hi = 0xA7C9 },   .{ .lo = 0xA7CB, .hi = 0xA7CC },   .{ .lo = 0xA7D0, .hi = 0xA7D0 },   .{ .lo = 0xA7D6, .hi = 0xA7D6 },
    .{ .lo = 0xA7D8, .hi = 0xA7D8 },   .{ .lo = 0xA7DA, .hi = 0xA7DA },   .{ .lo = 0xA7DC, .hi = 0xA7DC },   .{ .lo = 0xA7F5, .hi = 0xA7F5 },
    .{ .lo = 0xFF21, .hi = 0xFF3A },   .{ .lo = 0x10400, .hi = 0x10427 }, .{ .lo = 0x104B0, .hi = 0x104D3 }, .{ .lo = 0x10570, .hi = 0x1057A },
    .{ .lo = 0x1057C, .hi = 0x1058A }, .{ .lo = 0x1058C, .hi = 0x10592 }, .{ .lo = 0x10594, .hi = 0x10595 }, .{ .lo = 0x10C80, .hi = 0x10CB2 },
    .{ .lo = 0x10D50, .hi = 0x10D65 }, .{ .lo = 0x118A0, .hi = 0x118BF }, .{ .lo = 0x16E40, .hi = 0x16E5F }, .{ .lo = 0x1D400, .hi = 0x1D419 },
    .{ .lo = 0x1D434, .hi = 0x1D44D }, .{ .lo = 0x1D468, .hi = 0x1D481 }, .{ .lo = 0x1D49C, .hi = 0x1D49C }, .{ .lo = 0x1D49E, .hi = 0x1D49F },
    .{ .lo = 0x1D4A2, .hi = 0x1D4A2 }, .{ .lo = 0x1D4A5, .hi = 0x1D4A6 }, .{ .lo = 0x1D4A9, .hi = 0x1D4AC }, .{ .lo = 0x1D4AE, .hi = 0x1D4B5 },
    .{ .lo = 0x1D4D0, .hi = 0x1D4E9 }, .{ .lo = 0x1D504, .hi = 0x1D505 }, .{ .lo = 0x1D507, .hi = 0x1D50A }, .{ .lo = 0x1D50D, .hi = 0x1D514 },
    .{ .lo = 0x1D516, .hi = 0x1D51C }, .{ .lo = 0x1D538, .hi = 0x1D539 }, .{ .lo = 0x1D53B, .hi = 0x1D53E }, .{ .lo = 0x1D540, .hi = 0x1D544 },
    .{ .lo = 0x1D546, .hi = 0x1D546 }, .{ .lo = 0x1D54A, .hi = 0x1D550 }, .{ .lo = 0x1D56C, .hi = 0x1D585 }, .{ .lo = 0x1D5A0, .hi = 0x1D5B9 },
    .{ .lo = 0x1D5D4, .hi = 0x1D5ED }, .{ .lo = 0x1D608, .hi = 0x1D621 }, .{ .lo = 0x1D63C, .hi = 0x1D655 }, .{ .lo = 0x1D670, .hi = 0x1D689 },
    .{ .lo = 0x1D6A8, .hi = 0x1D6C0 }, .{ .lo = 0x1D6E2, .hi = 0x1D6FA }, .{ .lo = 0x1D71C, .hi = 0x1D734 }, .{ .lo = 0x1D756, .hi = 0x1D76E },
    .{ .lo = 0x1D790, .hi = 0x1D7A8 }, .{ .lo = 0x1D7CA, .hi = 0x1D7CA }, .{ .lo = 0x1E900, .hi = 0x1E921 },
};

const cat_Ll = [_]Range{
    .{ .lo = 0x0061, .hi = 0x007A },   .{ .lo = 0x00B5, .hi = 0x00B5 },   .{ .lo = 0x00DF, .hi = 0x00F6 },   .{ .lo = 0x00F8, .hi = 0x00FF },
    .{ .lo = 0x0101, .hi = 0x0101 },   .{ .lo = 0x0103, .hi = 0x0103 },   .{ .lo = 0x0105, .hi = 0x0105 },   .{ .lo = 0x0107, .hi = 0x0107 },
    .{ .lo = 0x0109, .hi = 0x0109 },   .{ .lo = 0x010B, .hi = 0x010B },   .{ .lo = 0x010D, .hi = 0x010D },   .{ .lo = 0x010F, .hi = 0x010F },
    .{ .lo = 0x0111, .hi = 0x0111 },   .{ .lo = 0x0113, .hi = 0x0113 },   .{ .lo = 0x0115, .hi = 0x0115 },   .{ .lo = 0x0117, .hi = 0x0117 },
    .{ .lo = 0x0119, .hi = 0x0119 },   .{ .lo = 0x011B, .hi = 0x011B },   .{ .lo = 0x011D, .hi = 0x011D },   .{ .lo = 0x011F, .hi = 0x011F },
    .{ .lo = 0x0121, .hi = 0x0121 },   .{ .lo = 0x0123, .hi = 0x0123 },   .{ .lo = 0x0125, .hi = 0x0125 },   .{ .lo = 0x0127, .hi = 0x0127 },
    .{ .lo = 0x0129, .hi = 0x0129 },   .{ .lo = 0x012B, .hi = 0x012B },   .{ .lo = 0x012D, .hi = 0x012D },   .{ .lo = 0x012F, .hi = 0x012F },
    .{ .lo = 0x0131, .hi = 0x0131 },   .{ .lo = 0x0133, .hi = 0x0133 },   .{ .lo = 0x0135, .hi = 0x0135 },   .{ .lo = 0x0137, .hi = 0x0138 },
    .{ .lo = 0x013A, .hi = 0x013A },   .{ .lo = 0x013C, .hi = 0x013C },   .{ .lo = 0x013E, .hi = 0x013E },   .{ .lo = 0x0140, .hi = 0x0140 },
    .{ .lo = 0x0142, .hi = 0x0142 },   .{ .lo = 0x0144, .hi = 0x0144 },   .{ .lo = 0x0146, .hi = 0x0146 },   .{ .lo = 0x0148, .hi = 0x0149 },
    .{ .lo = 0x014B, .hi = 0x014B },   .{ .lo = 0x014D, .hi = 0x014D },   .{ .lo = 0x014F, .hi = 0x014F },   .{ .lo = 0x0151, .hi = 0x0151 },
    .{ .lo = 0x0153, .hi = 0x0153 },   .{ .lo = 0x0155, .hi = 0x0155 },   .{ .lo = 0x0157, .hi = 0x0157 },   .{ .lo = 0x0159, .hi = 0x0159 },
    .{ .lo = 0x015B, .hi = 0x015B },   .{ .lo = 0x015D, .hi = 0x015D },   .{ .lo = 0x015F, .hi = 0x015F },   .{ .lo = 0x0161, .hi = 0x0161 },
    .{ .lo = 0x0163, .hi = 0x0163 },   .{ .lo = 0x0165, .hi = 0x0165 },   .{ .lo = 0x0167, .hi = 0x0167 },   .{ .lo = 0x0169, .hi = 0x0169 },
    .{ .lo = 0x016B, .hi = 0x016B },   .{ .lo = 0x016D, .hi = 0x016D },   .{ .lo = 0x016F, .hi = 0x016F },   .{ .lo = 0x0171, .hi = 0x0171 },
    .{ .lo = 0x0173, .hi = 0x0173 },   .{ .lo = 0x0175, .hi = 0x0175 },   .{ .lo = 0x0177, .hi = 0x0177 },   .{ .lo = 0x017A, .hi = 0x017A },
    .{ .lo = 0x017C, .hi = 0x017C },   .{ .lo = 0x017E, .hi = 0x0180 },   .{ .lo = 0x0183, .hi = 0x0183 },   .{ .lo = 0x0185, .hi = 0x0185 },
    .{ .lo = 0x0188, .hi = 0x0188 },   .{ .lo = 0x018C, .hi = 0x018D },   .{ .lo = 0x0192, .hi = 0x0192 },   .{ .lo = 0x0195, .hi = 0x0195 },
    .{ .lo = 0x0199, .hi = 0x019B },   .{ .lo = 0x019E, .hi = 0x019E },   .{ .lo = 0x01A1, .hi = 0x01A1 },   .{ .lo = 0x01A3, .hi = 0x01A3 },
    .{ .lo = 0x01A5, .hi = 0x01A5 },   .{ .lo = 0x01A8, .hi = 0x01A8 },   .{ .lo = 0x01AA, .hi = 0x01AB },   .{ .lo = 0x01AD, .hi = 0x01AD },
    .{ .lo = 0x01B0, .hi = 0x01B0 },   .{ .lo = 0x01B4, .hi = 0x01B4 },   .{ .lo = 0x01B6, .hi = 0x01B6 },   .{ .lo = 0x01B9, .hi = 0x01BA },
    .{ .lo = 0x01BD, .hi = 0x01BF },   .{ .lo = 0x01C6, .hi = 0x01C6 },   .{ .lo = 0x01C9, .hi = 0x01C9 },   .{ .lo = 0x01CC, .hi = 0x01CC },
    .{ .lo = 0x01CE, .hi = 0x01CE },   .{ .lo = 0x01D0, .hi = 0x01D0 },   .{ .lo = 0x01D2, .hi = 0x01D2 },   .{ .lo = 0x01D4, .hi = 0x01D4 },
    .{ .lo = 0x01D6, .hi = 0x01D6 },   .{ .lo = 0x01D8, .hi = 0x01D8 },   .{ .lo = 0x01DA, .hi = 0x01DA },   .{ .lo = 0x01DC, .hi = 0x01DD },
    .{ .lo = 0x01DF, .hi = 0x01DF },   .{ .lo = 0x01E1, .hi = 0x01E1 },   .{ .lo = 0x01E3, .hi = 0x01E3 },   .{ .lo = 0x01E5, .hi = 0x01E5 },
    .{ .lo = 0x01E7, .hi = 0x01E7 },   .{ .lo = 0x01E9, .hi = 0x01E9 },   .{ .lo = 0x01EB, .hi = 0x01EB },   .{ .lo = 0x01ED, .hi = 0x01ED },
    .{ .lo = 0x01EF, .hi = 0x01F0 },   .{ .lo = 0x01F3, .hi = 0x01F3 },   .{ .lo = 0x01F5, .hi = 0x01F5 },   .{ .lo = 0x01F9, .hi = 0x01F9 },
    .{ .lo = 0x01FB, .hi = 0x01FB },   .{ .lo = 0x01FD, .hi = 0x01FD },   .{ .lo = 0x01FF, .hi = 0x01FF },   .{ .lo = 0x0201, .hi = 0x0201 },
    .{ .lo = 0x0203, .hi = 0x0203 },   .{ .lo = 0x0205, .hi = 0x0205 },   .{ .lo = 0x0207, .hi = 0x0207 },   .{ .lo = 0x0209, .hi = 0x0209 },
    .{ .lo = 0x020B, .hi = 0x020B },   .{ .lo = 0x020D, .hi = 0x020D },   .{ .lo = 0x020F, .hi = 0x020F },   .{ .lo = 0x0211, .hi = 0x0211 },
    .{ .lo = 0x0213, .hi = 0x0213 },   .{ .lo = 0x0215, .hi = 0x0215 },   .{ .lo = 0x0217, .hi = 0x0217 },   .{ .lo = 0x0219, .hi = 0x0219 },
    .{ .lo = 0x021B, .hi = 0x021B },   .{ .lo = 0x021D, .hi = 0x021D },   .{ .lo = 0x021F, .hi = 0x021F },   .{ .lo = 0x0221, .hi = 0x0221 },
    .{ .lo = 0x0223, .hi = 0x0223 },   .{ .lo = 0x0225, .hi = 0x0225 },   .{ .lo = 0x0227, .hi = 0x0227 },   .{ .lo = 0x0229, .hi = 0x0229 },
    .{ .lo = 0x022B, .hi = 0x022B },   .{ .lo = 0x022D, .hi = 0x022D },   .{ .lo = 0x022F, .hi = 0x022F },   .{ .lo = 0x0231, .hi = 0x0231 },
    .{ .lo = 0x0233, .hi = 0x0239 },   .{ .lo = 0x023C, .hi = 0x023C },   .{ .lo = 0x023F, .hi = 0x0240 },   .{ .lo = 0x0242, .hi = 0x0242 },
    .{ .lo = 0x0247, .hi = 0x0247 },   .{ .lo = 0x0249, .hi = 0x0249 },   .{ .lo = 0x024B, .hi = 0x024B },   .{ .lo = 0x024D, .hi = 0x024D },
    .{ .lo = 0x024F, .hi = 0x0293 },   .{ .lo = 0x0295, .hi = 0x02AF },   .{ .lo = 0x0371, .hi = 0x0371 },   .{ .lo = 0x0373, .hi = 0x0373 },
    .{ .lo = 0x0377, .hi = 0x0377 },   .{ .lo = 0x037B, .hi = 0x037D },   .{ .lo = 0x0390, .hi = 0x0390 },   .{ .lo = 0x03AC, .hi = 0x03CE },
    .{ .lo = 0x03D0, .hi = 0x03D1 },   .{ .lo = 0x03D5, .hi = 0x03D7 },   .{ .lo = 0x03D9, .hi = 0x03D9 },   .{ .lo = 0x03DB, .hi = 0x03DB },
    .{ .lo = 0x03DD, .hi = 0x03DD },   .{ .lo = 0x03DF, .hi = 0x03DF },   .{ .lo = 0x03E1, .hi = 0x03E1 },   .{ .lo = 0x03E3, .hi = 0x03E3 },
    .{ .lo = 0x03E5, .hi = 0x03E5 },   .{ .lo = 0x03E7, .hi = 0x03E7 },   .{ .lo = 0x03E9, .hi = 0x03E9 },   .{ .lo = 0x03EB, .hi = 0x03EB },
    .{ .lo = 0x03ED, .hi = 0x03ED },   .{ .lo = 0x03EF, .hi = 0x03F3 },   .{ .lo = 0x03F5, .hi = 0x03F5 },   .{ .lo = 0x03F8, .hi = 0x03F8 },
    .{ .lo = 0x03FB, .hi = 0x03FC },   .{ .lo = 0x0430, .hi = 0x045F },   .{ .lo = 0x0461, .hi = 0x0461 },   .{ .lo = 0x0463, .hi = 0x0463 },
    .{ .lo = 0x0465, .hi = 0x0465 },   .{ .lo = 0x0467, .hi = 0x0467 },   .{ .lo = 0x0469, .hi = 0x0469 },   .{ .lo = 0x046B, .hi = 0x046B },
    .{ .lo = 0x046D, .hi = 0x046D },   .{ .lo = 0x046F, .hi = 0x046F },   .{ .lo = 0x0471, .hi = 0x0471 },   .{ .lo = 0x0473, .hi = 0x0473 },
    .{ .lo = 0x0475, .hi = 0x0475 },   .{ .lo = 0x0477, .hi = 0x0477 },   .{ .lo = 0x0479, .hi = 0x0479 },   .{ .lo = 0x047B, .hi = 0x047B },
    .{ .lo = 0x047D, .hi = 0x047D },   .{ .lo = 0x047F, .hi = 0x047F },   .{ .lo = 0x0481, .hi = 0x0481 },   .{ .lo = 0x048B, .hi = 0x048B },
    .{ .lo = 0x048D, .hi = 0x048D },   .{ .lo = 0x048F, .hi = 0x048F },   .{ .lo = 0x0491, .hi = 0x0491 },   .{ .lo = 0x0493, .hi = 0x0493 },
    .{ .lo = 0x0495, .hi = 0x0495 },   .{ .lo = 0x0497, .hi = 0x0497 },   .{ .lo = 0x0499, .hi = 0x0499 },   .{ .lo = 0x049B, .hi = 0x049B },
    .{ .lo = 0x049D, .hi = 0x049D },   .{ .lo = 0x049F, .hi = 0x049F },   .{ .lo = 0x04A1, .hi = 0x04A1 },   .{ .lo = 0x04A3, .hi = 0x04A3 },
    .{ .lo = 0x04A5, .hi = 0x04A5 },   .{ .lo = 0x04A7, .hi = 0x04A7 },   .{ .lo = 0x04A9, .hi = 0x04A9 },   .{ .lo = 0x04AB, .hi = 0x04AB },
    .{ .lo = 0x04AD, .hi = 0x04AD },   .{ .lo = 0x04AF, .hi = 0x04AF },   .{ .lo = 0x04B1, .hi = 0x04B1 },   .{ .lo = 0x04B3, .hi = 0x04B3 },
    .{ .lo = 0x04B5, .hi = 0x04B5 },   .{ .lo = 0x04B7, .hi = 0x04B7 },   .{ .lo = 0x04B9, .hi = 0x04B9 },   .{ .lo = 0x04BB, .hi = 0x04BB },
    .{ .lo = 0x04BD, .hi = 0x04BD },   .{ .lo = 0x04BF, .hi = 0x04BF },   .{ .lo = 0x04C2, .hi = 0x04C2 },   .{ .lo = 0x04C4, .hi = 0x04C4 },
    .{ .lo = 0x04C6, .hi = 0x04C6 },   .{ .lo = 0x04C8, .hi = 0x04C8 },   .{ .lo = 0x04CA, .hi = 0x04CA },   .{ .lo = 0x04CC, .hi = 0x04CC },
    .{ .lo = 0x04CE, .hi = 0x04CF },   .{ .lo = 0x04D1, .hi = 0x04D1 },   .{ .lo = 0x04D3, .hi = 0x04D3 },   .{ .lo = 0x04D5, .hi = 0x04D5 },
    .{ .lo = 0x04D7, .hi = 0x04D7 },   .{ .lo = 0x04D9, .hi = 0x04D9 },   .{ .lo = 0x04DB, .hi = 0x04DB },   .{ .lo = 0x04DD, .hi = 0x04DD },
    .{ .lo = 0x04DF, .hi = 0x04DF },   .{ .lo = 0x04E1, .hi = 0x04E1 },   .{ .lo = 0x04E3, .hi = 0x04E3 },   .{ .lo = 0x04E5, .hi = 0x04E5 },
    .{ .lo = 0x04E7, .hi = 0x04E7 },   .{ .lo = 0x04E9, .hi = 0x04E9 },   .{ .lo = 0x04EB, .hi = 0x04EB },   .{ .lo = 0x04ED, .hi = 0x04ED },
    .{ .lo = 0x04EF, .hi = 0x04EF },   .{ .lo = 0x04F1, .hi = 0x04F1 },   .{ .lo = 0x04F3, .hi = 0x04F3 },   .{ .lo = 0x04F5, .hi = 0x04F5 },
    .{ .lo = 0x04F7, .hi = 0x04F7 },   .{ .lo = 0x04F9, .hi = 0x04F9 },   .{ .lo = 0x04FB, .hi = 0x04FB },   .{ .lo = 0x04FD, .hi = 0x04FD },
    .{ .lo = 0x04FF, .hi = 0x04FF },   .{ .lo = 0x0501, .hi = 0x0501 },   .{ .lo = 0x0503, .hi = 0x0503 },   .{ .lo = 0x0505, .hi = 0x0505 },
    .{ .lo = 0x0507, .hi = 0x0507 },   .{ .lo = 0x0509, .hi = 0x0509 },   .{ .lo = 0x050B, .hi = 0x050B },   .{ .lo = 0x050D, .hi = 0x050D },
    .{ .lo = 0x050F, .hi = 0x050F },   .{ .lo = 0x0511, .hi = 0x0511 },   .{ .lo = 0x0513, .hi = 0x0513 },   .{ .lo = 0x0515, .hi = 0x0515 },
    .{ .lo = 0x0517, .hi = 0x0517 },   .{ .lo = 0x0519, .hi = 0x0519 },   .{ .lo = 0x051B, .hi = 0x051B },   .{ .lo = 0x051D, .hi = 0x051D },
    .{ .lo = 0x051F, .hi = 0x051F },   .{ .lo = 0x0521, .hi = 0x0521 },   .{ .lo = 0x0523, .hi = 0x0523 },   .{ .lo = 0x0525, .hi = 0x0525 },
    .{ .lo = 0x0527, .hi = 0x0527 },   .{ .lo = 0x0529, .hi = 0x0529 },   .{ .lo = 0x052B, .hi = 0x052B },   .{ .lo = 0x052D, .hi = 0x052D },
    .{ .lo = 0x052F, .hi = 0x052F },   .{ .lo = 0x0560, .hi = 0x0588 },   .{ .lo = 0x10D0, .hi = 0x10FA },   .{ .lo = 0x10FD, .hi = 0x10FF },
    .{ .lo = 0x13F8, .hi = 0x13FD },   .{ .lo = 0x1C80, .hi = 0x1C88 },   .{ .lo = 0x1C8A, .hi = 0x1C8A },   .{ .lo = 0x1D00, .hi = 0x1D2B },
    .{ .lo = 0x1D6B, .hi = 0x1D77 },   .{ .lo = 0x1D79, .hi = 0x1D9A },   .{ .lo = 0x1E01, .hi = 0x1E01 },   .{ .lo = 0x1E03, .hi = 0x1E03 },
    .{ .lo = 0x1E05, .hi = 0x1E05 },   .{ .lo = 0x1E07, .hi = 0x1E07 },   .{ .lo = 0x1E09, .hi = 0x1E09 },   .{ .lo = 0x1E0B, .hi = 0x1E0B },
    .{ .lo = 0x1E0D, .hi = 0x1E0D },   .{ .lo = 0x1E0F, .hi = 0x1E0F },   .{ .lo = 0x1E11, .hi = 0x1E11 },   .{ .lo = 0x1E13, .hi = 0x1E13 },
    .{ .lo = 0x1E15, .hi = 0x1E15 },   .{ .lo = 0x1E17, .hi = 0x1E17 },   .{ .lo = 0x1E19, .hi = 0x1E19 },   .{ .lo = 0x1E1B, .hi = 0x1E1B },
    .{ .lo = 0x1E1D, .hi = 0x1E1D },   .{ .lo = 0x1E1F, .hi = 0x1E1F },   .{ .lo = 0x1E21, .hi = 0x1E21 },   .{ .lo = 0x1E23, .hi = 0x1E23 },
    .{ .lo = 0x1E25, .hi = 0x1E25 },   .{ .lo = 0x1E27, .hi = 0x1E27 },   .{ .lo = 0x1E29, .hi = 0x1E29 },   .{ .lo = 0x1E2B, .hi = 0x1E2B },
    .{ .lo = 0x1E2D, .hi = 0x1E2D },   .{ .lo = 0x1E2F, .hi = 0x1E2F },   .{ .lo = 0x1E31, .hi = 0x1E31 },   .{ .lo = 0x1E33, .hi = 0x1E33 },
    .{ .lo = 0x1E35, .hi = 0x1E35 },   .{ .lo = 0x1E37, .hi = 0x1E37 },   .{ .lo = 0x1E39, .hi = 0x1E39 },   .{ .lo = 0x1E3B, .hi = 0x1E3B },
    .{ .lo = 0x1E3D, .hi = 0x1E3D },   .{ .lo = 0x1E3F, .hi = 0x1E3F },   .{ .lo = 0x1E41, .hi = 0x1E41 },   .{ .lo = 0x1E43, .hi = 0x1E43 },
    .{ .lo = 0x1E45, .hi = 0x1E45 },   .{ .lo = 0x1E47, .hi = 0x1E47 },   .{ .lo = 0x1E49, .hi = 0x1E49 },   .{ .lo = 0x1E4B, .hi = 0x1E4B },
    .{ .lo = 0x1E4D, .hi = 0x1E4D },   .{ .lo = 0x1E4F, .hi = 0x1E4F },   .{ .lo = 0x1E51, .hi = 0x1E51 },   .{ .lo = 0x1E53, .hi = 0x1E53 },
    .{ .lo = 0x1E55, .hi = 0x1E55 },   .{ .lo = 0x1E57, .hi = 0x1E57 },   .{ .lo = 0x1E59, .hi = 0x1E59 },   .{ .lo = 0x1E5B, .hi = 0x1E5B },
    .{ .lo = 0x1E5D, .hi = 0x1E5D },   .{ .lo = 0x1E5F, .hi = 0x1E5F },   .{ .lo = 0x1E61, .hi = 0x1E61 },   .{ .lo = 0x1E63, .hi = 0x1E63 },
    .{ .lo = 0x1E65, .hi = 0x1E65 },   .{ .lo = 0x1E67, .hi = 0x1E67 },   .{ .lo = 0x1E69, .hi = 0x1E69 },   .{ .lo = 0x1E6B, .hi = 0x1E6B },
    .{ .lo = 0x1E6D, .hi = 0x1E6D },   .{ .lo = 0x1E6F, .hi = 0x1E6F },   .{ .lo = 0x1E71, .hi = 0x1E71 },   .{ .lo = 0x1E73, .hi = 0x1E73 },
    .{ .lo = 0x1E75, .hi = 0x1E75 },   .{ .lo = 0x1E77, .hi = 0x1E77 },   .{ .lo = 0x1E79, .hi = 0x1E79 },   .{ .lo = 0x1E7B, .hi = 0x1E7B },
    .{ .lo = 0x1E7D, .hi = 0x1E7D },   .{ .lo = 0x1E7F, .hi = 0x1E7F },   .{ .lo = 0x1E81, .hi = 0x1E81 },   .{ .lo = 0x1E83, .hi = 0x1E83 },
    .{ .lo = 0x1E85, .hi = 0x1E85 },   .{ .lo = 0x1E87, .hi = 0x1E87 },   .{ .lo = 0x1E89, .hi = 0x1E89 },   .{ .lo = 0x1E8B, .hi = 0x1E8B },
    .{ .lo = 0x1E8D, .hi = 0x1E8D },   .{ .lo = 0x1E8F, .hi = 0x1E8F },   .{ .lo = 0x1E91, .hi = 0x1E91 },   .{ .lo = 0x1E93, .hi = 0x1E93 },
    .{ .lo = 0x1E95, .hi = 0x1E9D },   .{ .lo = 0x1E9F, .hi = 0x1E9F },   .{ .lo = 0x1EA1, .hi = 0x1EA1 },   .{ .lo = 0x1EA3, .hi = 0x1EA3 },
    .{ .lo = 0x1EA5, .hi = 0x1EA5 },   .{ .lo = 0x1EA7, .hi = 0x1EA7 },   .{ .lo = 0x1EA9, .hi = 0x1EA9 },   .{ .lo = 0x1EAB, .hi = 0x1EAB },
    .{ .lo = 0x1EAD, .hi = 0x1EAD },   .{ .lo = 0x1EAF, .hi = 0x1EAF },   .{ .lo = 0x1EB1, .hi = 0x1EB1 },   .{ .lo = 0x1EB3, .hi = 0x1EB3 },
    .{ .lo = 0x1EB5, .hi = 0x1EB5 },   .{ .lo = 0x1EB7, .hi = 0x1EB7 },   .{ .lo = 0x1EB9, .hi = 0x1EB9 },   .{ .lo = 0x1EBB, .hi = 0x1EBB },
    .{ .lo = 0x1EBD, .hi = 0x1EBD },   .{ .lo = 0x1EBF, .hi = 0x1EBF },   .{ .lo = 0x1EC1, .hi = 0x1EC1 },   .{ .lo = 0x1EC3, .hi = 0x1EC3 },
    .{ .lo = 0x1EC5, .hi = 0x1EC5 },   .{ .lo = 0x1EC7, .hi = 0x1EC7 },   .{ .lo = 0x1EC9, .hi = 0x1EC9 },   .{ .lo = 0x1ECB, .hi = 0x1ECB },
    .{ .lo = 0x1ECD, .hi = 0x1ECD },   .{ .lo = 0x1ECF, .hi = 0x1ECF },   .{ .lo = 0x1ED1, .hi = 0x1ED1 },   .{ .lo = 0x1ED3, .hi = 0x1ED3 },
    .{ .lo = 0x1ED5, .hi = 0x1ED5 },   .{ .lo = 0x1ED7, .hi = 0x1ED7 },   .{ .lo = 0x1ED9, .hi = 0x1ED9 },   .{ .lo = 0x1EDB, .hi = 0x1EDB },
    .{ .lo = 0x1EDD, .hi = 0x1EDD },   .{ .lo = 0x1EDF, .hi = 0x1EDF },   .{ .lo = 0x1EE1, .hi = 0x1EE1 },   .{ .lo = 0x1EE3, .hi = 0x1EE3 },
    .{ .lo = 0x1EE5, .hi = 0x1EE5 },   .{ .lo = 0x1EE7, .hi = 0x1EE7 },   .{ .lo = 0x1EE9, .hi = 0x1EE9 },   .{ .lo = 0x1EEB, .hi = 0x1EEB },
    .{ .lo = 0x1EED, .hi = 0x1EED },   .{ .lo = 0x1EEF, .hi = 0x1EEF },   .{ .lo = 0x1EF1, .hi = 0x1EF1 },   .{ .lo = 0x1EF3, .hi = 0x1EF3 },
    .{ .lo = 0x1EF5, .hi = 0x1EF5 },   .{ .lo = 0x1EF7, .hi = 0x1EF7 },   .{ .lo = 0x1EF9, .hi = 0x1EF9 },   .{ .lo = 0x1EFB, .hi = 0x1EFB },
    .{ .lo = 0x1EFD, .hi = 0x1EFD },   .{ .lo = 0x1EFF, .hi = 0x1F07 },   .{ .lo = 0x1F10, .hi = 0x1F15 },   .{ .lo = 0x1F20, .hi = 0x1F27 },
    .{ .lo = 0x1F30, .hi = 0x1F37 },   .{ .lo = 0x1F40, .hi = 0x1F45 },   .{ .lo = 0x1F50, .hi = 0x1F57 },   .{ .lo = 0x1F60, .hi = 0x1F67 },
    .{ .lo = 0x1F70, .hi = 0x1F7D },   .{ .lo = 0x1F80, .hi = 0x1F87 },   .{ .lo = 0x1F90, .hi = 0x1F97 },   .{ .lo = 0x1FA0, .hi = 0x1FA7 },
    .{ .lo = 0x1FB0, .hi = 0x1FB4 },   .{ .lo = 0x1FB6, .hi = 0x1FB7 },   .{ .lo = 0x1FBE, .hi = 0x1FBE },   .{ .lo = 0x1FC2, .hi = 0x1FC4 },
    .{ .lo = 0x1FC6, .hi = 0x1FC7 },   .{ .lo = 0x1FD0, .hi = 0x1FD3 },   .{ .lo = 0x1FD6, .hi = 0x1FD7 },   .{ .lo = 0x1FE0, .hi = 0x1FE7 },
    .{ .lo = 0x1FF2, .hi = 0x1FF4 },   .{ .lo = 0x1FF6, .hi = 0x1FF7 },   .{ .lo = 0x210A, .hi = 0x210A },   .{ .lo = 0x210E, .hi = 0x210F },
    .{ .lo = 0x2113, .hi = 0x2113 },   .{ .lo = 0x212F, .hi = 0x212F },   .{ .lo = 0x2134, .hi = 0x2134 },   .{ .lo = 0x2139, .hi = 0x2139 },
    .{ .lo = 0x213C, .hi = 0x213D },   .{ .lo = 0x2146, .hi = 0x2149 },   .{ .lo = 0x214E, .hi = 0x214E },   .{ .lo = 0x2184, .hi = 0x2184 },
    .{ .lo = 0x2C30, .hi = 0x2C5F },   .{ .lo = 0x2C61, .hi = 0x2C61 },   .{ .lo = 0x2C65, .hi = 0x2C66 },   .{ .lo = 0x2C68, .hi = 0x2C68 },
    .{ .lo = 0x2C6A, .hi = 0x2C6A },   .{ .lo = 0x2C6C, .hi = 0x2C6C },   .{ .lo = 0x2C71, .hi = 0x2C71 },   .{ .lo = 0x2C73, .hi = 0x2C74 },
    .{ .lo = 0x2C76, .hi = 0x2C7B },   .{ .lo = 0x2C81, .hi = 0x2C81 },   .{ .lo = 0x2C83, .hi = 0x2C83 },   .{ .lo = 0x2C85, .hi = 0x2C85 },
    .{ .lo = 0x2C87, .hi = 0x2C87 },   .{ .lo = 0x2C89, .hi = 0x2C89 },   .{ .lo = 0x2C8B, .hi = 0x2C8B },   .{ .lo = 0x2C8D, .hi = 0x2C8D },
    .{ .lo = 0x2C8F, .hi = 0x2C8F },   .{ .lo = 0x2C91, .hi = 0x2C91 },   .{ .lo = 0x2C93, .hi = 0x2C93 },   .{ .lo = 0x2C95, .hi = 0x2C95 },
    .{ .lo = 0x2C97, .hi = 0x2C97 },   .{ .lo = 0x2C99, .hi = 0x2C99 },   .{ .lo = 0x2C9B, .hi = 0x2C9B },   .{ .lo = 0x2C9D, .hi = 0x2C9D },
    .{ .lo = 0x2C9F, .hi = 0x2C9F },   .{ .lo = 0x2CA1, .hi = 0x2CA1 },   .{ .lo = 0x2CA3, .hi = 0x2CA3 },   .{ .lo = 0x2CA5, .hi = 0x2CA5 },
    .{ .lo = 0x2CA7, .hi = 0x2CA7 },   .{ .lo = 0x2CA9, .hi = 0x2CA9 },   .{ .lo = 0x2CAB, .hi = 0x2CAB },   .{ .lo = 0x2CAD, .hi = 0x2CAD },
    .{ .lo = 0x2CAF, .hi = 0x2CAF },   .{ .lo = 0x2CB1, .hi = 0x2CB1 },   .{ .lo = 0x2CB3, .hi = 0x2CB3 },   .{ .lo = 0x2CB5, .hi = 0x2CB5 },
    .{ .lo = 0x2CB7, .hi = 0x2CB7 },   .{ .lo = 0x2CB9, .hi = 0x2CB9 },   .{ .lo = 0x2CBB, .hi = 0x2CBB },   .{ .lo = 0x2CBD, .hi = 0x2CBD },
    .{ .lo = 0x2CBF, .hi = 0x2CBF },   .{ .lo = 0x2CC1, .hi = 0x2CC1 },   .{ .lo = 0x2CC3, .hi = 0x2CC3 },   .{ .lo = 0x2CC5, .hi = 0x2CC5 },
    .{ .lo = 0x2CC7, .hi = 0x2CC7 },   .{ .lo = 0x2CC9, .hi = 0x2CC9 },   .{ .lo = 0x2CCB, .hi = 0x2CCB },   .{ .lo = 0x2CCD, .hi = 0x2CCD },
    .{ .lo = 0x2CCF, .hi = 0x2CCF },   .{ .lo = 0x2CD1, .hi = 0x2CD1 },   .{ .lo = 0x2CD3, .hi = 0x2CD3 },   .{ .lo = 0x2CD5, .hi = 0x2CD5 },
    .{ .lo = 0x2CD7, .hi = 0x2CD7 },   .{ .lo = 0x2CD9, .hi = 0x2CD9 },   .{ .lo = 0x2CDB, .hi = 0x2CDB },   .{ .lo = 0x2CDD, .hi = 0x2CDD },
    .{ .lo = 0x2CDF, .hi = 0x2CDF },   .{ .lo = 0x2CE1, .hi = 0x2CE1 },   .{ .lo = 0x2CE3, .hi = 0x2CE4 },   .{ .lo = 0x2CEC, .hi = 0x2CEC },
    .{ .lo = 0x2CEE, .hi = 0x2CEE },   .{ .lo = 0x2CF3, .hi = 0x2CF3 },   .{ .lo = 0x2D00, .hi = 0x2D25 },   .{ .lo = 0x2D27, .hi = 0x2D27 },
    .{ .lo = 0x2D2D, .hi = 0x2D2D },   .{ .lo = 0xA641, .hi = 0xA641 },   .{ .lo = 0xA643, .hi = 0xA643 },   .{ .lo = 0xA645, .hi = 0xA645 },
    .{ .lo = 0xA647, .hi = 0xA647 },   .{ .lo = 0xA649, .hi = 0xA649 },   .{ .lo = 0xA64B, .hi = 0xA64B },   .{ .lo = 0xA64D, .hi = 0xA64D },
    .{ .lo = 0xA64F, .hi = 0xA64F },   .{ .lo = 0xA651, .hi = 0xA651 },   .{ .lo = 0xA653, .hi = 0xA653 },   .{ .lo = 0xA655, .hi = 0xA655 },
    .{ .lo = 0xA657, .hi = 0xA657 },   .{ .lo = 0xA659, .hi = 0xA659 },   .{ .lo = 0xA65B, .hi = 0xA65B },   .{ .lo = 0xA65D, .hi = 0xA65D },
    .{ .lo = 0xA65F, .hi = 0xA65F },   .{ .lo = 0xA661, .hi = 0xA661 },   .{ .lo = 0xA663, .hi = 0xA663 },   .{ .lo = 0xA665, .hi = 0xA665 },
    .{ .lo = 0xA667, .hi = 0xA667 },   .{ .lo = 0xA669, .hi = 0xA669 },   .{ .lo = 0xA66B, .hi = 0xA66B },   .{ .lo = 0xA66D, .hi = 0xA66D },
    .{ .lo = 0xA681, .hi = 0xA681 },   .{ .lo = 0xA683, .hi = 0xA683 },   .{ .lo = 0xA685, .hi = 0xA685 },   .{ .lo = 0xA687, .hi = 0xA687 },
    .{ .lo = 0xA689, .hi = 0xA689 },   .{ .lo = 0xA68B, .hi = 0xA68B },   .{ .lo = 0xA68D, .hi = 0xA68D },   .{ .lo = 0xA68F, .hi = 0xA68F },
    .{ .lo = 0xA691, .hi = 0xA691 },   .{ .lo = 0xA693, .hi = 0xA693 },   .{ .lo = 0xA695, .hi = 0xA695 },   .{ .lo = 0xA697, .hi = 0xA697 },
    .{ .lo = 0xA699, .hi = 0xA699 },   .{ .lo = 0xA69B, .hi = 0xA69B },   .{ .lo = 0xA723, .hi = 0xA723 },   .{ .lo = 0xA725, .hi = 0xA725 },
    .{ .lo = 0xA727, .hi = 0xA727 },   .{ .lo = 0xA729, .hi = 0xA729 },   .{ .lo = 0xA72B, .hi = 0xA72B },   .{ .lo = 0xA72D, .hi = 0xA72D },
    .{ .lo = 0xA72F, .hi = 0xA731 },   .{ .lo = 0xA733, .hi = 0xA733 },   .{ .lo = 0xA735, .hi = 0xA735 },   .{ .lo = 0xA737, .hi = 0xA737 },
    .{ .lo = 0xA739, .hi = 0xA739 },   .{ .lo = 0xA73B, .hi = 0xA73B },   .{ .lo = 0xA73D, .hi = 0xA73D },   .{ .lo = 0xA73F, .hi = 0xA73F },
    .{ .lo = 0xA741, .hi = 0xA741 },   .{ .lo = 0xA743, .hi = 0xA743 },   .{ .lo = 0xA745, .hi = 0xA745 },   .{ .lo = 0xA747, .hi = 0xA747 },
    .{ .lo = 0xA749, .hi = 0xA749 },   .{ .lo = 0xA74B, .hi = 0xA74B },   .{ .lo = 0xA74D, .hi = 0xA74D },   .{ .lo = 0xA74F, .hi = 0xA74F },
    .{ .lo = 0xA751, .hi = 0xA751 },   .{ .lo = 0xA753, .hi = 0xA753 },   .{ .lo = 0xA755, .hi = 0xA755 },   .{ .lo = 0xA757, .hi = 0xA757 },
    .{ .lo = 0xA759, .hi = 0xA759 },   .{ .lo = 0xA75B, .hi = 0xA75B },   .{ .lo = 0xA75D, .hi = 0xA75D },   .{ .lo = 0xA75F, .hi = 0xA75F },
    .{ .lo = 0xA761, .hi = 0xA761 },   .{ .lo = 0xA763, .hi = 0xA763 },   .{ .lo = 0xA765, .hi = 0xA765 },   .{ .lo = 0xA767, .hi = 0xA767 },
    .{ .lo = 0xA769, .hi = 0xA769 },   .{ .lo = 0xA76B, .hi = 0xA76B },   .{ .lo = 0xA76D, .hi = 0xA76D },   .{ .lo = 0xA76F, .hi = 0xA76F },
    .{ .lo = 0xA771, .hi = 0xA778 },   .{ .lo = 0xA77A, .hi = 0xA77A },   .{ .lo = 0xA77C, .hi = 0xA77C },   .{ .lo = 0xA77F, .hi = 0xA77F },
    .{ .lo = 0xA781, .hi = 0xA781 },   .{ .lo = 0xA783, .hi = 0xA783 },   .{ .lo = 0xA785, .hi = 0xA785 },   .{ .lo = 0xA787, .hi = 0xA787 },
    .{ .lo = 0xA78C, .hi = 0xA78C },   .{ .lo = 0xA78E, .hi = 0xA78E },   .{ .lo = 0xA791, .hi = 0xA791 },   .{ .lo = 0xA793, .hi = 0xA795 },
    .{ .lo = 0xA797, .hi = 0xA797 },   .{ .lo = 0xA799, .hi = 0xA799 },   .{ .lo = 0xA79B, .hi = 0xA79B },   .{ .lo = 0xA79D, .hi = 0xA79D },
    .{ .lo = 0xA79F, .hi = 0xA79F },   .{ .lo = 0xA7A1, .hi = 0xA7A1 },   .{ .lo = 0xA7A3, .hi = 0xA7A3 },   .{ .lo = 0xA7A5, .hi = 0xA7A5 },
    .{ .lo = 0xA7A7, .hi = 0xA7A7 },   .{ .lo = 0xA7A9, .hi = 0xA7A9 },   .{ .lo = 0xA7AF, .hi = 0xA7AF },   .{ .lo = 0xA7B5, .hi = 0xA7B5 },
    .{ .lo = 0xA7B7, .hi = 0xA7B7 },   .{ .lo = 0xA7B9, .hi = 0xA7B9 },   .{ .lo = 0xA7BB, .hi = 0xA7BB },   .{ .lo = 0xA7BD, .hi = 0xA7BD },
    .{ .lo = 0xA7BF, .hi = 0xA7BF },   .{ .lo = 0xA7C1, .hi = 0xA7C1 },   .{ .lo = 0xA7C3, .hi = 0xA7C3 },   .{ .lo = 0xA7C8, .hi = 0xA7C8 },
    .{ .lo = 0xA7CA, .hi = 0xA7CA },   .{ .lo = 0xA7CD, .hi = 0xA7CD },   .{ .lo = 0xA7D1, .hi = 0xA7D1 },   .{ .lo = 0xA7D3, .hi = 0xA7D3 },
    .{ .lo = 0xA7D5, .hi = 0xA7D5 },   .{ .lo = 0xA7D7, .hi = 0xA7D7 },   .{ .lo = 0xA7D9, .hi = 0xA7D9 },   .{ .lo = 0xA7DB, .hi = 0xA7DB },
    .{ .lo = 0xA7F6, .hi = 0xA7F6 },   .{ .lo = 0xA7FA, .hi = 0xA7FA },   .{ .lo = 0xAB30, .hi = 0xAB5A },   .{ .lo = 0xAB60, .hi = 0xAB68 },
    .{ .lo = 0xAB70, .hi = 0xABBF },   .{ .lo = 0xFB00, .hi = 0xFB06 },   .{ .lo = 0xFB13, .hi = 0xFB17 },   .{ .lo = 0xFF41, .hi = 0xFF5A },
    .{ .lo = 0x10428, .hi = 0x1044F }, .{ .lo = 0x104D8, .hi = 0x104FB }, .{ .lo = 0x10597, .hi = 0x105A1 }, .{ .lo = 0x105A3, .hi = 0x105B1 },
    .{ .lo = 0x105B3, .hi = 0x105B9 }, .{ .lo = 0x105BB, .hi = 0x105BC }, .{ .lo = 0x10CC0, .hi = 0x10CF2 }, .{ .lo = 0x10D70, .hi = 0x10D85 },
    .{ .lo = 0x118C0, .hi = 0x118DF }, .{ .lo = 0x16E60, .hi = 0x16E7F }, .{ .lo = 0x1D41A, .hi = 0x1D433 }, .{ .lo = 0x1D44E, .hi = 0x1D454 },
    .{ .lo = 0x1D456, .hi = 0x1D467 }, .{ .lo = 0x1D482, .hi = 0x1D49B }, .{ .lo = 0x1D4B6, .hi = 0x1D4B9 }, .{ .lo = 0x1D4BB, .hi = 0x1D4BB },
    .{ .lo = 0x1D4BD, .hi = 0x1D4C3 }, .{ .lo = 0x1D4C5, .hi = 0x1D4CF }, .{ .lo = 0x1D4EA, .hi = 0x1D503 }, .{ .lo = 0x1D51E, .hi = 0x1D537 },
    .{ .lo = 0x1D552, .hi = 0x1D56B }, .{ .lo = 0x1D586, .hi = 0x1D59F }, .{ .lo = 0x1D5BA, .hi = 0x1D5D3 }, .{ .lo = 0x1D5EE, .hi = 0x1D607 },
    .{ .lo = 0x1D622, .hi = 0x1D63B }, .{ .lo = 0x1D656, .hi = 0x1D66F }, .{ .lo = 0x1D68A, .hi = 0x1D6A5 }, .{ .lo = 0x1D6C2, .hi = 0x1D6DA },
    .{ .lo = 0x1D6DC, .hi = 0x1D6E1 }, .{ .lo = 0x1D6FC, .hi = 0x1D714 }, .{ .lo = 0x1D716, .hi = 0x1D71B }, .{ .lo = 0x1D736, .hi = 0x1D74E },
    .{ .lo = 0x1D750, .hi = 0x1D755 }, .{ .lo = 0x1D770, .hi = 0x1D788 }, .{ .lo = 0x1D78A, .hi = 0x1D78F }, .{ .lo = 0x1D7AA, .hi = 0x1D7C2 },
    .{ .lo = 0x1D7C4, .hi = 0x1D7C9 }, .{ .lo = 0x1D7CB, .hi = 0x1D7CB }, .{ .lo = 0x1DF00, .hi = 0x1DF09 }, .{ .lo = 0x1DF0B, .hi = 0x1DF1E },
    .{ .lo = 0x1DF25, .hi = 0x1DF2A }, .{ .lo = 0x1E922, .hi = 0x1E943 },
};

const cat_Lt = [_]Range{
    .{ .lo = 0x01C5, .hi = 0x01C5 }, .{ .lo = 0x01C8, .hi = 0x01C8 }, .{ .lo = 0x01CB, .hi = 0x01CB }, .{ .lo = 0x01F2, .hi = 0x01F2 },
    .{ .lo = 0x1F88, .hi = 0x1F8F }, .{ .lo = 0x1F98, .hi = 0x1F9F }, .{ .lo = 0x1FA8, .hi = 0x1FAF }, .{ .lo = 0x1FBC, .hi = 0x1FBC },
    .{ .lo = 0x1FCC, .hi = 0x1FCC }, .{ .lo = 0x1FFC, .hi = 0x1FFC },
};

const cat_Lm = [_]Range{
    .{ .lo = 0x02B0, .hi = 0x02C1 },   .{ .lo = 0x02C6, .hi = 0x02D1 },   .{ .lo = 0x02E0, .hi = 0x02E4 },   .{ .lo = 0x02EC, .hi = 0x02EC },
    .{ .lo = 0x02EE, .hi = 0x02EE },   .{ .lo = 0x0374, .hi = 0x0374 },   .{ .lo = 0x037A, .hi = 0x037A },   .{ .lo = 0x0559, .hi = 0x0559 },
    .{ .lo = 0x0640, .hi = 0x0640 },   .{ .lo = 0x06E5, .hi = 0x06E6 },   .{ .lo = 0x07F4, .hi = 0x07F5 },   .{ .lo = 0x07FA, .hi = 0x07FA },
    .{ .lo = 0x081A, .hi = 0x081A },   .{ .lo = 0x0824, .hi = 0x0824 },   .{ .lo = 0x0828, .hi = 0x0828 },   .{ .lo = 0x08C9, .hi = 0x08C9 },
    .{ .lo = 0x0971, .hi = 0x0971 },   .{ .lo = 0x0E46, .hi = 0x0E46 },   .{ .lo = 0x0EC6, .hi = 0x0EC6 },   .{ .lo = 0x10FC, .hi = 0x10FC },
    .{ .lo = 0x17D7, .hi = 0x17D7 },   .{ .lo = 0x1843, .hi = 0x1843 },   .{ .lo = 0x1AA7, .hi = 0x1AA7 },   .{ .lo = 0x1C78, .hi = 0x1C7D },
    .{ .lo = 0x1D2C, .hi = 0x1D6A },   .{ .lo = 0x1D78, .hi = 0x1D78 },   .{ .lo = 0x1D9B, .hi = 0x1DBF },   .{ .lo = 0x2071, .hi = 0x2071 },
    .{ .lo = 0x207F, .hi = 0x207F },   .{ .lo = 0x2090, .hi = 0x209C },   .{ .lo = 0x2C7C, .hi = 0x2C7D },   .{ .lo = 0x2D6F, .hi = 0x2D6F },
    .{ .lo = 0x2E2F, .hi = 0x2E2F },   .{ .lo = 0x3005, .hi = 0x3005 },   .{ .lo = 0x3031, .hi = 0x3035 },   .{ .lo = 0x303B, .hi = 0x303B },
    .{ .lo = 0x309D, .hi = 0x309E },   .{ .lo = 0x30FC, .hi = 0x30FE },   .{ .lo = 0xA015, .hi = 0xA015 },   .{ .lo = 0xA4F8, .hi = 0xA4FD },
    .{ .lo = 0xA60C, .hi = 0xA60C },   .{ .lo = 0xA67F, .hi = 0xA67F },   .{ .lo = 0xA69C, .hi = 0xA69D },   .{ .lo = 0xA717, .hi = 0xA71F },
    .{ .lo = 0xA770, .hi = 0xA770 },   .{ .lo = 0xA788, .hi = 0xA788 },   .{ .lo = 0xA7F2, .hi = 0xA7F4 },   .{ .lo = 0xA7F8, .hi = 0xA7F9 },
    .{ .lo = 0xA9CF, .hi = 0xA9CF },   .{ .lo = 0xA9E6, .hi = 0xA9E6 },   .{ .lo = 0xAA70, .hi = 0xAA70 },   .{ .lo = 0xAADD, .hi = 0xAADD },
    .{ .lo = 0xAAF3, .hi = 0xAAF4 },   .{ .lo = 0xAB5C, .hi = 0xAB5F },   .{ .lo = 0xAB69, .hi = 0xAB69 },   .{ .lo = 0xFF70, .hi = 0xFF70 },
    .{ .lo = 0xFF9E, .hi = 0xFF9F },   .{ .lo = 0x10780, .hi = 0x10785 }, .{ .lo = 0x10787, .hi = 0x107B0 }, .{ .lo = 0x107B2, .hi = 0x107BA },
    .{ .lo = 0x10D4E, .hi = 0x10D4E }, .{ .lo = 0x10D6F, .hi = 0x10D6F }, .{ .lo = 0x16B40, .hi = 0x16B43 }, .{ .lo = 0x16D40, .hi = 0x16D42 },
    .{ .lo = 0x16D6B, .hi = 0x16D6C }, .{ .lo = 0x16F93, .hi = 0x16F9F }, .{ .lo = 0x16FE0, .hi = 0x16FE1 }, .{ .lo = 0x16FE3, .hi = 0x16FE3 },
    .{ .lo = 0x1AFF0, .hi = 0x1AFF3 }, .{ .lo = 0x1AFF5, .hi = 0x1AFFB }, .{ .lo = 0x1AFFD, .hi = 0x1AFFE }, .{ .lo = 0x1E030, .hi = 0x1E06D },
    .{ .lo = 0x1E137, .hi = 0x1E13D }, .{ .lo = 0x1E4EB, .hi = 0x1E4EB }, .{ .lo = 0x1E94B, .hi = 0x1E94B },
};

const cat_Lo = [_]Range{
    .{ .lo = 0x00AA, .hi = 0x00AA },   .{ .lo = 0x00BA, .hi = 0x00BA },   .{ .lo = 0x01BB, .hi = 0x01BB },   .{ .lo = 0x01C0, .hi = 0x01C3 },
    .{ .lo = 0x0294, .hi = 0x0294 },   .{ .lo = 0x05D0, .hi = 0x05EA },   .{ .lo = 0x05EF, .hi = 0x05F2 },   .{ .lo = 0x0620, .hi = 0x063F },
    .{ .lo = 0x0641, .hi = 0x064A },   .{ .lo = 0x066E, .hi = 0x066F },   .{ .lo = 0x0671, .hi = 0x06D3 },   .{ .lo = 0x06D5, .hi = 0x06D5 },
    .{ .lo = 0x06EE, .hi = 0x06EF },   .{ .lo = 0x06FA, .hi = 0x06FC },   .{ .lo = 0x06FF, .hi = 0x06FF },   .{ .lo = 0x0710, .hi = 0x0710 },
    .{ .lo = 0x0712, .hi = 0x072F },   .{ .lo = 0x074D, .hi = 0x07A5 },   .{ .lo = 0x07B1, .hi = 0x07B1 },   .{ .lo = 0x07CA, .hi = 0x07EA },
    .{ .lo = 0x0800, .hi = 0x0815 },   .{ .lo = 0x0840, .hi = 0x0858 },   .{ .lo = 0x0860, .hi = 0x086A },   .{ .lo = 0x0870, .hi = 0x0887 },
    .{ .lo = 0x0889, .hi = 0x088E },   .{ .lo = 0x08A0, .hi = 0x08C8 },   .{ .lo = 0x0904, .hi = 0x0939 },   .{ .lo = 0x093D, .hi = 0x093D },
    .{ .lo = 0x0950, .hi = 0x0950 },   .{ .lo = 0x0958, .hi = 0x0961 },   .{ .lo = 0x0972, .hi = 0x0980 },   .{ .lo = 0x0985, .hi = 0x098C },
    .{ .lo = 0x098F, .hi = 0x0990 },   .{ .lo = 0x0993, .hi = 0x09A8 },   .{ .lo = 0x09AA, .hi = 0x09B0 },   .{ .lo = 0x09B2, .hi = 0x09B2 },
    .{ .lo = 0x09B6, .hi = 0x09B9 },   .{ .lo = 0x09BD, .hi = 0x09BD },   .{ .lo = 0x09CE, .hi = 0x09CE },   .{ .lo = 0x09DC, .hi = 0x09DD },
    .{ .lo = 0x09DF, .hi = 0x09E1 },   .{ .lo = 0x09F0, .hi = 0x09F1 },   .{ .lo = 0x09FC, .hi = 0x09FC },   .{ .lo = 0x0A05, .hi = 0x0A0A },
    .{ .lo = 0x0A0F, .hi = 0x0A10 },   .{ .lo = 0x0A13, .hi = 0x0A28 },   .{ .lo = 0x0A2A, .hi = 0x0A30 },   .{ .lo = 0x0A32, .hi = 0x0A33 },
    .{ .lo = 0x0A35, .hi = 0x0A36 },   .{ .lo = 0x0A38, .hi = 0x0A39 },   .{ .lo = 0x0A59, .hi = 0x0A5C },   .{ .lo = 0x0A5E, .hi = 0x0A5E },
    .{ .lo = 0x0A72, .hi = 0x0A74 },   .{ .lo = 0x0A85, .hi = 0x0A8D },   .{ .lo = 0x0A8F, .hi = 0x0A91 },   .{ .lo = 0x0A93, .hi = 0x0AA8 },
    .{ .lo = 0x0AAA, .hi = 0x0AB0 },   .{ .lo = 0x0AB2, .hi = 0x0AB3 },   .{ .lo = 0x0AB5, .hi = 0x0AB9 },   .{ .lo = 0x0ABD, .hi = 0x0ABD },
    .{ .lo = 0x0AD0, .hi = 0x0AD0 },   .{ .lo = 0x0AE0, .hi = 0x0AE1 },   .{ .lo = 0x0AF9, .hi = 0x0AF9 },   .{ .lo = 0x0B05, .hi = 0x0B0C },
    .{ .lo = 0x0B0F, .hi = 0x0B10 },   .{ .lo = 0x0B13, .hi = 0x0B28 },   .{ .lo = 0x0B2A, .hi = 0x0B30 },   .{ .lo = 0x0B32, .hi = 0x0B33 },
    .{ .lo = 0x0B35, .hi = 0x0B39 },   .{ .lo = 0x0B3D, .hi = 0x0B3D },   .{ .lo = 0x0B5C, .hi = 0x0B5D },   .{ .lo = 0x0B5F, .hi = 0x0B61 },
    .{ .lo = 0x0B71, .hi = 0x0B71 },   .{ .lo = 0x0B83, .hi = 0x0B83 },   .{ .lo = 0x0B85, .hi = 0x0B8A },   .{ .lo = 0x0B8E, .hi = 0x0B90 },
    .{ .lo = 0x0B92, .hi = 0x0B95 },   .{ .lo = 0x0B99, .hi = 0x0B9A },   .{ .lo = 0x0B9C, .hi = 0x0B9C },   .{ .lo = 0x0B9E, .hi = 0x0B9F },
    .{ .lo = 0x0BA3, .hi = 0x0BA4 },   .{ .lo = 0x0BA8, .hi = 0x0BAA },   .{ .lo = 0x0BAE, .hi = 0x0BB9 },   .{ .lo = 0x0BD0, .hi = 0x0BD0 },
    .{ .lo = 0x0C05, .hi = 0x0C0C },   .{ .lo = 0x0C0E, .hi = 0x0C10 },   .{ .lo = 0x0C12, .hi = 0x0C28 },   .{ .lo = 0x0C2A, .hi = 0x0C39 },
    .{ .lo = 0x0C3D, .hi = 0x0C3D },   .{ .lo = 0x0C58, .hi = 0x0C5A },   .{ .lo = 0x0C5D, .hi = 0x0C5D },   .{ .lo = 0x0C60, .hi = 0x0C61 },
    .{ .lo = 0x0C80, .hi = 0x0C80 },   .{ .lo = 0x0C85, .hi = 0x0C8C },   .{ .lo = 0x0C8E, .hi = 0x0C90 },   .{ .lo = 0x0C92, .hi = 0x0CA8 },
    .{ .lo = 0x0CAA, .hi = 0x0CB3 },   .{ .lo = 0x0CB5, .hi = 0x0CB9 },   .{ .lo = 0x0CBD, .hi = 0x0CBD },   .{ .lo = 0x0CDD, .hi = 0x0CDE },
    .{ .lo = 0x0CE0, .hi = 0x0CE1 },   .{ .lo = 0x0CF1, .hi = 0x0CF2 },   .{ .lo = 0x0D04, .hi = 0x0D0C },   .{ .lo = 0x0D0E, .hi = 0x0D10 },
    .{ .lo = 0x0D12, .hi = 0x0D3A },   .{ .lo = 0x0D3D, .hi = 0x0D3D },   .{ .lo = 0x0D4E, .hi = 0x0D4E },   .{ .lo = 0x0D54, .hi = 0x0D56 },
    .{ .lo = 0x0D5F, .hi = 0x0D61 },   .{ .lo = 0x0D7A, .hi = 0x0D7F },   .{ .lo = 0x0D85, .hi = 0x0D96 },   .{ .lo = 0x0D9A, .hi = 0x0DB1 },
    .{ .lo = 0x0DB3, .hi = 0x0DBB },   .{ .lo = 0x0DBD, .hi = 0x0DBD },   .{ .lo = 0x0DC0, .hi = 0x0DC6 },   .{ .lo = 0x0E01, .hi = 0x0E30 },
    .{ .lo = 0x0E32, .hi = 0x0E33 },   .{ .lo = 0x0E40, .hi = 0x0E45 },   .{ .lo = 0x0E81, .hi = 0x0E82 },   .{ .lo = 0x0E84, .hi = 0x0E84 },
    .{ .lo = 0x0E86, .hi = 0x0E8A },   .{ .lo = 0x0E8C, .hi = 0x0EA3 },   .{ .lo = 0x0EA5, .hi = 0x0EA5 },   .{ .lo = 0x0EA7, .hi = 0x0EB0 },
    .{ .lo = 0x0EB2, .hi = 0x0EB3 },   .{ .lo = 0x0EBD, .hi = 0x0EBD },   .{ .lo = 0x0EC0, .hi = 0x0EC4 },   .{ .lo = 0x0EDC, .hi = 0x0EDF },
    .{ .lo = 0x0F00, .hi = 0x0F00 },   .{ .lo = 0x0F40, .hi = 0x0F47 },   .{ .lo = 0x0F49, .hi = 0x0F6C },   .{ .lo = 0x0F88, .hi = 0x0F8C },
    .{ .lo = 0x1000, .hi = 0x102A },   .{ .lo = 0x103F, .hi = 0x103F },   .{ .lo = 0x1050, .hi = 0x1055 },   .{ .lo = 0x105A, .hi = 0x105D },
    .{ .lo = 0x1061, .hi = 0x1061 },   .{ .lo = 0x1065, .hi = 0x1066 },   .{ .lo = 0x106E, .hi = 0x1070 },   .{ .lo = 0x1075, .hi = 0x1081 },
    .{ .lo = 0x108E, .hi = 0x108E },   .{ .lo = 0x1100, .hi = 0x1248 },   .{ .lo = 0x124A, .hi = 0x124D },   .{ .lo = 0x1250, .hi = 0x1256 },
    .{ .lo = 0x1258, .hi = 0x1258 },   .{ .lo = 0x125A, .hi = 0x125D },   .{ .lo = 0x1260, .hi = 0x1288 },   .{ .lo = 0x128A, .hi = 0x128D },
    .{ .lo = 0x1290, .hi = 0x12B0 },   .{ .lo = 0x12B2, .hi = 0x12B5 },   .{ .lo = 0x12B8, .hi = 0x12BE },   .{ .lo = 0x12C0, .hi = 0x12C0 },
    .{ .lo = 0x12C2, .hi = 0x12C5 },   .{ .lo = 0x12C8, .hi = 0x12D6 },   .{ .lo = 0x12D8, .hi = 0x1310 },   .{ .lo = 0x1312, .hi = 0x1315 },
    .{ .lo = 0x1318, .hi = 0x135A },   .{ .lo = 0x1380, .hi = 0x138F },   .{ .lo = 0x1401, .hi = 0x166C },   .{ .lo = 0x166F, .hi = 0x167F },
    .{ .lo = 0x1681, .hi = 0x169A },   .{ .lo = 0x16A0, .hi = 0x16EA },   .{ .lo = 0x16F1, .hi = 0x16F8 },   .{ .lo = 0x1700, .hi = 0x1711 },
    .{ .lo = 0x171F, .hi = 0x1731 },   .{ .lo = 0x1740, .hi = 0x1751 },   .{ .lo = 0x1760, .hi = 0x176C },   .{ .lo = 0x176E, .hi = 0x1770 },
    .{ .lo = 0x1780, .hi = 0x17B3 },   .{ .lo = 0x17DC, .hi = 0x17DC },   .{ .lo = 0x1820, .hi = 0x1842 },   .{ .lo = 0x1844, .hi = 0x1878 },
    .{ .lo = 0x1880, .hi = 0x1884 },   .{ .lo = 0x1887, .hi = 0x18A8 },   .{ .lo = 0x18AA, .hi = 0x18AA },   .{ .lo = 0x18B0, .hi = 0x18F5 },
    .{ .lo = 0x1900, .hi = 0x191E },   .{ .lo = 0x1950, .hi = 0x196D },   .{ .lo = 0x1970, .hi = 0x1974 },   .{ .lo = 0x1980, .hi = 0x19AB },
    .{ .lo = 0x19B0, .hi = 0x19C9 },   .{ .lo = 0x1A00, .hi = 0x1A16 },   .{ .lo = 0x1A20, .hi = 0x1A54 },   .{ .lo = 0x1B05, .hi = 0x1B33 },
    .{ .lo = 0x1B45, .hi = 0x1B4C },   .{ .lo = 0x1B83, .hi = 0x1BA0 },   .{ .lo = 0x1BAE, .hi = 0x1BAF },   .{ .lo = 0x1BBA, .hi = 0x1BE5 },
    .{ .lo = 0x1C00, .hi = 0x1C23 },   .{ .lo = 0x1C4D, .hi = 0x1C4F },   .{ .lo = 0x1C5A, .hi = 0x1C77 },   .{ .lo = 0x1CE9, .hi = 0x1CEC },
    .{ .lo = 0x1CEE, .hi = 0x1CF3 },   .{ .lo = 0x1CF5, .hi = 0x1CF6 },   .{ .lo = 0x1CFA, .hi = 0x1CFA },   .{ .lo = 0x2135, .hi = 0x2138 },
    .{ .lo = 0x2D30, .hi = 0x2D67 },   .{ .lo = 0x2D80, .hi = 0x2D96 },   .{ .lo = 0x2DA0, .hi = 0x2DA6 },   .{ .lo = 0x2DA8, .hi = 0x2DAE },
    .{ .lo = 0x2DB0, .hi = 0x2DB6 },   .{ .lo = 0x2DB8, .hi = 0x2DBE },   .{ .lo = 0x2DC0, .hi = 0x2DC6 },   .{ .lo = 0x2DC8, .hi = 0x2DCE },
    .{ .lo = 0x2DD0, .hi = 0x2DD6 },   .{ .lo = 0x2DD8, .hi = 0x2DDE },   .{ .lo = 0x3006, .hi = 0x3006 },   .{ .lo = 0x303C, .hi = 0x303C },
    .{ .lo = 0x3041, .hi = 0x3096 },   .{ .lo = 0x309F, .hi = 0x309F },   .{ .lo = 0x30A1, .hi = 0x30FA },   .{ .lo = 0x30FF, .hi = 0x30FF },
    .{ .lo = 0x3105, .hi = 0x312F },   .{ .lo = 0x3131, .hi = 0x318E },   .{ .lo = 0x31A0, .hi = 0x31BF },   .{ .lo = 0x31F0, .hi = 0x31FF },
    .{ .lo = 0x3400, .hi = 0x4DBF },   .{ .lo = 0x4E00, .hi = 0xA014 },   .{ .lo = 0xA016, .hi = 0xA48C },   .{ .lo = 0xA4D0, .hi = 0xA4F7 },
    .{ .lo = 0xA500, .hi = 0xA60B },   .{ .lo = 0xA610, .hi = 0xA61F },   .{ .lo = 0xA62A, .hi = 0xA62B },   .{ .lo = 0xA66E, .hi = 0xA66E },
    .{ .lo = 0xA6A0, .hi = 0xA6E5 },   .{ .lo = 0xA78F, .hi = 0xA78F },   .{ .lo = 0xA7F7, .hi = 0xA7F7 },   .{ .lo = 0xA7FB, .hi = 0xA801 },
    .{ .lo = 0xA803, .hi = 0xA805 },   .{ .lo = 0xA807, .hi = 0xA80A },   .{ .lo = 0xA80C, .hi = 0xA822 },   .{ .lo = 0xA840, .hi = 0xA873 },
    .{ .lo = 0xA882, .hi = 0xA8B3 },   .{ .lo = 0xA8F2, .hi = 0xA8F7 },   .{ .lo = 0xA8FB, .hi = 0xA8FB },   .{ .lo = 0xA8FD, .hi = 0xA8FE },
    .{ .lo = 0xA90A, .hi = 0xA925 },   .{ .lo = 0xA930, .hi = 0xA946 },   .{ .lo = 0xA960, .hi = 0xA97C },   .{ .lo = 0xA984, .hi = 0xA9B2 },
    .{ .lo = 0xA9E0, .hi = 0xA9E4 },   .{ .lo = 0xA9E7, .hi = 0xA9EF },   .{ .lo = 0xA9FA, .hi = 0xA9FE },   .{ .lo = 0xAA00, .hi = 0xAA28 },
    .{ .lo = 0xAA40, .hi = 0xAA42 },   .{ .lo = 0xAA44, .hi = 0xAA4B },   .{ .lo = 0xAA60, .hi = 0xAA6F },   .{ .lo = 0xAA71, .hi = 0xAA76 },
    .{ .lo = 0xAA7A, .hi = 0xAA7A },   .{ .lo = 0xAA7E, .hi = 0xAAAF },   .{ .lo = 0xAAB1, .hi = 0xAAB1 },   .{ .lo = 0xAAB5, .hi = 0xAAB6 },
    .{ .lo = 0xAAB9, .hi = 0xAABD },   .{ .lo = 0xAAC0, .hi = 0xAAC0 },   .{ .lo = 0xAAC2, .hi = 0xAAC2 },   .{ .lo = 0xAADB, .hi = 0xAADC },
    .{ .lo = 0xAAE0, .hi = 0xAAEA },   .{ .lo = 0xAAF2, .hi = 0xAAF2 },   .{ .lo = 0xAB01, .hi = 0xAB06 },   .{ .lo = 0xAB09, .hi = 0xAB0E },
    .{ .lo = 0xAB11, .hi = 0xAB16 },   .{ .lo = 0xAB20, .hi = 0xAB26 },   .{ .lo = 0xAB28, .hi = 0xAB2E },   .{ .lo = 0xABC0, .hi = 0xABE2 },
    .{ .lo = 0xAC00, .hi = 0xD7A3 },   .{ .lo = 0xD7B0, .hi = 0xD7C6 },   .{ .lo = 0xD7CB, .hi = 0xD7FB },   .{ .lo = 0xF900, .hi = 0xFA6D },
    .{ .lo = 0xFA70, .hi = 0xFAD9 },   .{ .lo = 0xFB1D, .hi = 0xFB1D },   .{ .lo = 0xFB1F, .hi = 0xFB28 },   .{ .lo = 0xFB2A, .hi = 0xFB36 },
    .{ .lo = 0xFB38, .hi = 0xFB3C },   .{ .lo = 0xFB3E, .hi = 0xFB3E },   .{ .lo = 0xFB40, .hi = 0xFB41 },   .{ .lo = 0xFB43, .hi = 0xFB44 },
    .{ .lo = 0xFB46, .hi = 0xFBB1 },   .{ .lo = 0xFBD3, .hi = 0xFD3D },   .{ .lo = 0xFD50, .hi = 0xFD8F },   .{ .lo = 0xFD92, .hi = 0xFDC7 },
    .{ .lo = 0xFDF0, .hi = 0xFDFB },   .{ .lo = 0xFE70, .hi = 0xFE74 },   .{ .lo = 0xFE76, .hi = 0xFEFC },   .{ .lo = 0xFF66, .hi = 0xFF6F },
    .{ .lo = 0xFF71, .hi = 0xFF9D },   .{ .lo = 0xFFA0, .hi = 0xFFBE },   .{ .lo = 0xFFC2, .hi = 0xFFC7 },   .{ .lo = 0xFFCA, .hi = 0xFFCF },
    .{ .lo = 0xFFD2, .hi = 0xFFD7 },   .{ .lo = 0xFFDA, .hi = 0xFFDC },   .{ .lo = 0x10000, .hi = 0x1000B }, .{ .lo = 0x1000D, .hi = 0x10026 },
    .{ .lo = 0x10028, .hi = 0x1003A }, .{ .lo = 0x1003C, .hi = 0x1003D }, .{ .lo = 0x1003F, .hi = 0x1004D }, .{ .lo = 0x10050, .hi = 0x1005D },
    .{ .lo = 0x10080, .hi = 0x100FA }, .{ .lo = 0x10280, .hi = 0x1029C }, .{ .lo = 0x102A0, .hi = 0x102D0 }, .{ .lo = 0x10300, .hi = 0x1031F },
    .{ .lo = 0x1032D, .hi = 0x10340 }, .{ .lo = 0x10342, .hi = 0x10349 }, .{ .lo = 0x10350, .hi = 0x10375 }, .{ .lo = 0x10380, .hi = 0x1039D },
    .{ .lo = 0x103A0, .hi = 0x103C3 }, .{ .lo = 0x103C8, .hi = 0x103CF }, .{ .lo = 0x10450, .hi = 0x1049D }, .{ .lo = 0x10500, .hi = 0x10527 },
    .{ .lo = 0x10530, .hi = 0x10563 }, .{ .lo = 0x105C0, .hi = 0x105F3 }, .{ .lo = 0x10600, .hi = 0x10736 }, .{ .lo = 0x10740, .hi = 0x10755 },
    .{ .lo = 0x10760, .hi = 0x10767 }, .{ .lo = 0x10800, .hi = 0x10805 }, .{ .lo = 0x10808, .hi = 0x10808 }, .{ .lo = 0x1080A, .hi = 0x10835 },
    .{ .lo = 0x10837, .hi = 0x10838 }, .{ .lo = 0x1083C, .hi = 0x1083C }, .{ .lo = 0x1083F, .hi = 0x10855 }, .{ .lo = 0x10860, .hi = 0x10876 },
    .{ .lo = 0x10880, .hi = 0x1089E }, .{ .lo = 0x108E0, .hi = 0x108F2 }, .{ .lo = 0x108F4, .hi = 0x108F5 }, .{ .lo = 0x10900, .hi = 0x10915 },
    .{ .lo = 0x10920, .hi = 0x10939 }, .{ .lo = 0x10980, .hi = 0x109B7 }, .{ .lo = 0x109BE, .hi = 0x109BF }, .{ .lo = 0x10A00, .hi = 0x10A00 },
    .{ .lo = 0x10A10, .hi = 0x10A13 }, .{ .lo = 0x10A15, .hi = 0x10A17 }, .{ .lo = 0x10A19, .hi = 0x10A35 }, .{ .lo = 0x10A60, .hi = 0x10A7C },
    .{ .lo = 0x10A80, .hi = 0x10A9C }, .{ .lo = 0x10AC0, .hi = 0x10AC7 }, .{ .lo = 0x10AC9, .hi = 0x10AE4 }, .{ .lo = 0x10B00, .hi = 0x10B35 },
    .{ .lo = 0x10B40, .hi = 0x10B55 }, .{ .lo = 0x10B60, .hi = 0x10B72 }, .{ .lo = 0x10B80, .hi = 0x10B91 }, .{ .lo = 0x10C00, .hi = 0x10C48 },
    .{ .lo = 0x10D00, .hi = 0x10D23 }, .{ .lo = 0x10D4A, .hi = 0x10D4D }, .{ .lo = 0x10D4F, .hi = 0x10D4F }, .{ .lo = 0x10E80, .hi = 0x10EA9 },
    .{ .lo = 0x10EB0, .hi = 0x10EB1 }, .{ .lo = 0x10EC2, .hi = 0x10EC4 }, .{ .lo = 0x10F00, .hi = 0x10F1C }, .{ .lo = 0x10F27, .hi = 0x10F27 },
    .{ .lo = 0x10F30, .hi = 0x10F45 }, .{ .lo = 0x10F70, .hi = 0x10F81 }, .{ .lo = 0x10FB0, .hi = 0x10FC4 }, .{ .lo = 0x10FE0, .hi = 0x10FF6 },
    .{ .lo = 0x11003, .hi = 0x11037 }, .{ .lo = 0x11071, .hi = 0x11072 }, .{ .lo = 0x11075, .hi = 0x11075 }, .{ .lo = 0x11083, .hi = 0x110AF },
    .{ .lo = 0x110D0, .hi = 0x110E8 }, .{ .lo = 0x11103, .hi = 0x11126 }, .{ .lo = 0x11144, .hi = 0x11144 }, .{ .lo = 0x11147, .hi = 0x11147 },
    .{ .lo = 0x11150, .hi = 0x11172 }, .{ .lo = 0x11176, .hi = 0x11176 }, .{ .lo = 0x11183, .hi = 0x111B2 }, .{ .lo = 0x111C1, .hi = 0x111C4 },
    .{ .lo = 0x111DA, .hi = 0x111DA }, .{ .lo = 0x111DC, .hi = 0x111DC }, .{ .lo = 0x11200, .hi = 0x11211 }, .{ .lo = 0x11213, .hi = 0x1122B },
    .{ .lo = 0x1123F, .hi = 0x11240 }, .{ .lo = 0x11280, .hi = 0x11286 }, .{ .lo = 0x11288, .hi = 0x11288 }, .{ .lo = 0x1128A, .hi = 0x1128D },
    .{ .lo = 0x1128F, .hi = 0x1129D }, .{ .lo = 0x1129F, .hi = 0x112A8 }, .{ .lo = 0x112B0, .hi = 0x112DE }, .{ .lo = 0x11305, .hi = 0x1130C },
    .{ .lo = 0x1130F, .hi = 0x11310 }, .{ .lo = 0x11313, .hi = 0x11328 }, .{ .lo = 0x1132A, .hi = 0x11330 }, .{ .lo = 0x11332, .hi = 0x11333 },
    .{ .lo = 0x11335, .hi = 0x11339 }, .{ .lo = 0x1133D, .hi = 0x1133D }, .{ .lo = 0x11350, .hi = 0x11350 }, .{ .lo = 0x1135D, .hi = 0x11361 },
    .{ .lo = 0x11380, .hi = 0x11389 }, .{ .lo = 0x1138B, .hi = 0x1138B }, .{ .lo = 0x1138E, .hi = 0x1138E }, .{ .lo = 0x11390, .hi = 0x113B5 },
    .{ .lo = 0x113B7, .hi = 0x113B7 }, .{ .lo = 0x113D1, .hi = 0x113D1 }, .{ .lo = 0x113D3, .hi = 0x113D3 }, .{ .lo = 0x11400, .hi = 0x11434 },
    .{ .lo = 0x11447, .hi = 0x1144A }, .{ .lo = 0x1145F, .hi = 0x11461 }, .{ .lo = 0x11480, .hi = 0x114AF }, .{ .lo = 0x114C4, .hi = 0x114C5 },
    .{ .lo = 0x114C7, .hi = 0x114C7 }, .{ .lo = 0x11580, .hi = 0x115AE }, .{ .lo = 0x115D8, .hi = 0x115DB }, .{ .lo = 0x11600, .hi = 0x1162F },
    .{ .lo = 0x11644, .hi = 0x11644 }, .{ .lo = 0x11680, .hi = 0x116AA }, .{ .lo = 0x116B8, .hi = 0x116B8 }, .{ .lo = 0x11700, .hi = 0x1171A },
    .{ .lo = 0x11740, .hi = 0x11746 }, .{ .lo = 0x11800, .hi = 0x1182B }, .{ .lo = 0x118FF, .hi = 0x11906 }, .{ .lo = 0x11909, .hi = 0x11909 },
    .{ .lo = 0x1190C, .hi = 0x11913 }, .{ .lo = 0x11915, .hi = 0x11916 }, .{ .lo = 0x11918, .hi = 0x1192F }, .{ .lo = 0x1193F, .hi = 0x1193F },
    .{ .lo = 0x11941, .hi = 0x11941 }, .{ .lo = 0x119A0, .hi = 0x119A7 }, .{ .lo = 0x119AA, .hi = 0x119D0 }, .{ .lo = 0x119E1, .hi = 0x119E1 },
    .{ .lo = 0x119E3, .hi = 0x119E3 }, .{ .lo = 0x11A00, .hi = 0x11A00 }, .{ .lo = 0x11A0B, .hi = 0x11A32 }, .{ .lo = 0x11A3A, .hi = 0x11A3A },
    .{ .lo = 0x11A50, .hi = 0x11A50 }, .{ .lo = 0x11A5C, .hi = 0x11A89 }, .{ .lo = 0x11A9D, .hi = 0x11A9D }, .{ .lo = 0x11AB0, .hi = 0x11AF8 },
    .{ .lo = 0x11BC0, .hi = 0x11BE0 }, .{ .lo = 0x11C00, .hi = 0x11C08 }, .{ .lo = 0x11C0A, .hi = 0x11C2E }, .{ .lo = 0x11C40, .hi = 0x11C40 },
    .{ .lo = 0x11C72, .hi = 0x11C8F }, .{ .lo = 0x11D00, .hi = 0x11D06 }, .{ .lo = 0x11D08, .hi = 0x11D09 }, .{ .lo = 0x11D0B, .hi = 0x11D30 },
    .{ .lo = 0x11D46, .hi = 0x11D46 }, .{ .lo = 0x11D60, .hi = 0x11D65 }, .{ .lo = 0x11D67, .hi = 0x11D68 }, .{ .lo = 0x11D6A, .hi = 0x11D89 },
    .{ .lo = 0x11D98, .hi = 0x11D98 }, .{ .lo = 0x11EE0, .hi = 0x11EF2 }, .{ .lo = 0x11F02, .hi = 0x11F02 }, .{ .lo = 0x11F04, .hi = 0x11F10 },
    .{ .lo = 0x11F12, .hi = 0x11F33 }, .{ .lo = 0x11FB0, .hi = 0x11FB0 }, .{ .lo = 0x12000, .hi = 0x12399 }, .{ .lo = 0x12480, .hi = 0x12543 },
    .{ .lo = 0x12F90, .hi = 0x12FF0 }, .{ .lo = 0x13000, .hi = 0x1342F }, .{ .lo = 0x13441, .hi = 0x13446 }, .{ .lo = 0x13460, .hi = 0x143FA },
    .{ .lo = 0x14400, .hi = 0x14646 }, .{ .lo = 0x16100, .hi = 0x1611D }, .{ .lo = 0x16800, .hi = 0x16A38 }, .{ .lo = 0x16A40, .hi = 0x16A5E },
    .{ .lo = 0x16A70, .hi = 0x16ABE }, .{ .lo = 0x16AD0, .hi = 0x16AED }, .{ .lo = 0x16B00, .hi = 0x16B2F }, .{ .lo = 0x16B63, .hi = 0x16B77 },
    .{ .lo = 0x16B7D, .hi = 0x16B8F }, .{ .lo = 0x16D43, .hi = 0x16D6A }, .{ .lo = 0x16F00, .hi = 0x16F4A }, .{ .lo = 0x16F50, .hi = 0x16F50 },
    .{ .lo = 0x17000, .hi = 0x187F7 }, .{ .lo = 0x18800, .hi = 0x18CD5 }, .{ .lo = 0x18CFF, .hi = 0x18D08 }, .{ .lo = 0x1B000, .hi = 0x1B122 },
    .{ .lo = 0x1B132, .hi = 0x1B132 }, .{ .lo = 0x1B150, .hi = 0x1B152 }, .{ .lo = 0x1B155, .hi = 0x1B155 }, .{ .lo = 0x1B164, .hi = 0x1B167 },
    .{ .lo = 0x1B170, .hi = 0x1B2FB }, .{ .lo = 0x1BC00, .hi = 0x1BC6A }, .{ .lo = 0x1BC70, .hi = 0x1BC7C }, .{ .lo = 0x1BC80, .hi = 0x1BC88 },
    .{ .lo = 0x1BC90, .hi = 0x1BC99 }, .{ .lo = 0x1DF0A, .hi = 0x1DF0A }, .{ .lo = 0x1E100, .hi = 0x1E12C }, .{ .lo = 0x1E14E, .hi = 0x1E14E },
    .{ .lo = 0x1E290, .hi = 0x1E2AD }, .{ .lo = 0x1E2C0, .hi = 0x1E2EB }, .{ .lo = 0x1E4D0, .hi = 0x1E4EA }, .{ .lo = 0x1E5D0, .hi = 0x1E5ED },
    .{ .lo = 0x1E5F0, .hi = 0x1E5F0 }, .{ .lo = 0x1E7E0, .hi = 0x1E7E6 }, .{ .lo = 0x1E7E8, .hi = 0x1E7EB }, .{ .lo = 0x1E7ED, .hi = 0x1E7EE },
    .{ .lo = 0x1E7F0, .hi = 0x1E7FE }, .{ .lo = 0x1E800, .hi = 0x1E8C4 }, .{ .lo = 0x1EE00, .hi = 0x1EE03 }, .{ .lo = 0x1EE05, .hi = 0x1EE1F },
    .{ .lo = 0x1EE21, .hi = 0x1EE22 }, .{ .lo = 0x1EE24, .hi = 0x1EE24 }, .{ .lo = 0x1EE27, .hi = 0x1EE27 }, .{ .lo = 0x1EE29, .hi = 0x1EE32 },
    .{ .lo = 0x1EE34, .hi = 0x1EE37 }, .{ .lo = 0x1EE39, .hi = 0x1EE39 }, .{ .lo = 0x1EE3B, .hi = 0x1EE3B }, .{ .lo = 0x1EE42, .hi = 0x1EE42 },
    .{ .lo = 0x1EE47, .hi = 0x1EE47 }, .{ .lo = 0x1EE49, .hi = 0x1EE49 }, .{ .lo = 0x1EE4B, .hi = 0x1EE4B }, .{ .lo = 0x1EE4D, .hi = 0x1EE4F },
    .{ .lo = 0x1EE51, .hi = 0x1EE52 }, .{ .lo = 0x1EE54, .hi = 0x1EE54 }, .{ .lo = 0x1EE57, .hi = 0x1EE57 }, .{ .lo = 0x1EE59, .hi = 0x1EE59 },
    .{ .lo = 0x1EE5B, .hi = 0x1EE5B }, .{ .lo = 0x1EE5D, .hi = 0x1EE5D }, .{ .lo = 0x1EE5F, .hi = 0x1EE5F }, .{ .lo = 0x1EE61, .hi = 0x1EE62 },
    .{ .lo = 0x1EE64, .hi = 0x1EE64 }, .{ .lo = 0x1EE67, .hi = 0x1EE6A }, .{ .lo = 0x1EE6C, .hi = 0x1EE72 }, .{ .lo = 0x1EE74, .hi = 0x1EE77 },
    .{ .lo = 0x1EE79, .hi = 0x1EE7C }, .{ .lo = 0x1EE7E, .hi = 0x1EE7E }, .{ .lo = 0x1EE80, .hi = 0x1EE89 }, .{ .lo = 0x1EE8B, .hi = 0x1EE9B },
    .{ .lo = 0x1EEA1, .hi = 0x1EEA3 }, .{ .lo = 0x1EEA5, .hi = 0x1EEA9 }, .{ .lo = 0x1EEAB, .hi = 0x1EEBB }, .{ .lo = 0x20000, .hi = 0x2A6DF },
    .{ .lo = 0x2A700, .hi = 0x2B739 }, .{ .lo = 0x2B740, .hi = 0x2B81D }, .{ .lo = 0x2B820, .hi = 0x2CEA1 }, .{ .lo = 0x2CEB0, .hi = 0x2EBE0 },
    .{ .lo = 0x2EBF0, .hi = 0x2EE5D }, .{ .lo = 0x2F800, .hi = 0x2FA1D }, .{ .lo = 0x30000, .hi = 0x3134A }, .{ .lo = 0x31350, .hi = 0x323AF },
};

const cat_Mn = [_]Range{
    .{ .lo = 0x0300, .hi = 0x036F },   .{ .lo = 0x0483, .hi = 0x0487 },   .{ .lo = 0x0591, .hi = 0x05BD },   .{ .lo = 0x05BF, .hi = 0x05BF },
    .{ .lo = 0x05C1, .hi = 0x05C2 },   .{ .lo = 0x05C4, .hi = 0x05C5 },   .{ .lo = 0x05C7, .hi = 0x05C7 },   .{ .lo = 0x0610, .hi = 0x061A },
    .{ .lo = 0x064B, .hi = 0x065F },   .{ .lo = 0x0670, .hi = 0x0670 },   .{ .lo = 0x06D6, .hi = 0x06DC },   .{ .lo = 0x06DF, .hi = 0x06E4 },
    .{ .lo = 0x06E7, .hi = 0x06E8 },   .{ .lo = 0x06EA, .hi = 0x06ED },   .{ .lo = 0x0711, .hi = 0x0711 },   .{ .lo = 0x0730, .hi = 0x074A },
    .{ .lo = 0x07A6, .hi = 0x07B0 },   .{ .lo = 0x07EB, .hi = 0x07F3 },   .{ .lo = 0x07FD, .hi = 0x07FD },   .{ .lo = 0x0816, .hi = 0x0819 },
    .{ .lo = 0x081B, .hi = 0x0823 },   .{ .lo = 0x0825, .hi = 0x0827 },   .{ .lo = 0x0829, .hi = 0x082D },   .{ .lo = 0x0859, .hi = 0x085B },
    .{ .lo = 0x0897, .hi = 0x089F },   .{ .lo = 0x08CA, .hi = 0x08E1 },   .{ .lo = 0x08E3, .hi = 0x0902 },   .{ .lo = 0x093A, .hi = 0x093A },
    .{ .lo = 0x093C, .hi = 0x093C },   .{ .lo = 0x0941, .hi = 0x0948 },   .{ .lo = 0x094D, .hi = 0x094D },   .{ .lo = 0x0951, .hi = 0x0957 },
    .{ .lo = 0x0962, .hi = 0x0963 },   .{ .lo = 0x0981, .hi = 0x0981 },   .{ .lo = 0x09BC, .hi = 0x09BC },   .{ .lo = 0x09C1, .hi = 0x09C4 },
    .{ .lo = 0x09CD, .hi = 0x09CD },   .{ .lo = 0x09E2, .hi = 0x09E3 },   .{ .lo = 0x09FE, .hi = 0x09FE },   .{ .lo = 0x0A01, .hi = 0x0A02 },
    .{ .lo = 0x0A3C, .hi = 0x0A3C },   .{ .lo = 0x0A41, .hi = 0x0A42 },   .{ .lo = 0x0A47, .hi = 0x0A48 },   .{ .lo = 0x0A4B, .hi = 0x0A4D },
    .{ .lo = 0x0A51, .hi = 0x0A51 },   .{ .lo = 0x0A70, .hi = 0x0A71 },   .{ .lo = 0x0A75, .hi = 0x0A75 },   .{ .lo = 0x0A81, .hi = 0x0A82 },
    .{ .lo = 0x0ABC, .hi = 0x0ABC },   .{ .lo = 0x0AC1, .hi = 0x0AC5 },   .{ .lo = 0x0AC7, .hi = 0x0AC8 },   .{ .lo = 0x0ACD, .hi = 0x0ACD },
    .{ .lo = 0x0AE2, .hi = 0x0AE3 },   .{ .lo = 0x0AFA, .hi = 0x0AFF },   .{ .lo = 0x0B01, .hi = 0x0B01 },   .{ .lo = 0x0B3C, .hi = 0x0B3C },
    .{ .lo = 0x0B3F, .hi = 0x0B3F },   .{ .lo = 0x0B41, .hi = 0x0B44 },   .{ .lo = 0x0B4D, .hi = 0x0B4D },   .{ .lo = 0x0B55, .hi = 0x0B56 },
    .{ .lo = 0x0B62, .hi = 0x0B63 },   .{ .lo = 0x0B82, .hi = 0x0B82 },   .{ .lo = 0x0BC0, .hi = 0x0BC0 },   .{ .lo = 0x0BCD, .hi = 0x0BCD },
    .{ .lo = 0x0C00, .hi = 0x0C00 },   .{ .lo = 0x0C04, .hi = 0x0C04 },   .{ .lo = 0x0C3C, .hi = 0x0C3C },   .{ .lo = 0x0C3E, .hi = 0x0C40 },
    .{ .lo = 0x0C46, .hi = 0x0C48 },   .{ .lo = 0x0C4A, .hi = 0x0C4D },   .{ .lo = 0x0C55, .hi = 0x0C56 },   .{ .lo = 0x0C62, .hi = 0x0C63 },
    .{ .lo = 0x0C81, .hi = 0x0C81 },   .{ .lo = 0x0CBC, .hi = 0x0CBC },   .{ .lo = 0x0CBF, .hi = 0x0CBF },   .{ .lo = 0x0CC6, .hi = 0x0CC6 },
    .{ .lo = 0x0CCC, .hi = 0x0CCD },   .{ .lo = 0x0CE2, .hi = 0x0CE3 },   .{ .lo = 0x0D00, .hi = 0x0D01 },   .{ .lo = 0x0D3B, .hi = 0x0D3C },
    .{ .lo = 0x0D41, .hi = 0x0D44 },   .{ .lo = 0x0D4D, .hi = 0x0D4D },   .{ .lo = 0x0D62, .hi = 0x0D63 },   .{ .lo = 0x0D81, .hi = 0x0D81 },
    .{ .lo = 0x0DCA, .hi = 0x0DCA },   .{ .lo = 0x0DD2, .hi = 0x0DD4 },   .{ .lo = 0x0DD6, .hi = 0x0DD6 },   .{ .lo = 0x0E31, .hi = 0x0E31 },
    .{ .lo = 0x0E34, .hi = 0x0E3A },   .{ .lo = 0x0E47, .hi = 0x0E4E },   .{ .lo = 0x0EB1, .hi = 0x0EB1 },   .{ .lo = 0x0EB4, .hi = 0x0EBC },
    .{ .lo = 0x0EC8, .hi = 0x0ECE },   .{ .lo = 0x0F18, .hi = 0x0F19 },   .{ .lo = 0x0F35, .hi = 0x0F35 },   .{ .lo = 0x0F37, .hi = 0x0F37 },
    .{ .lo = 0x0F39, .hi = 0x0F39 },   .{ .lo = 0x0F71, .hi = 0x0F7E },   .{ .lo = 0x0F80, .hi = 0x0F84 },   .{ .lo = 0x0F86, .hi = 0x0F87 },
    .{ .lo = 0x0F8D, .hi = 0x0F97 },   .{ .lo = 0x0F99, .hi = 0x0FBC },   .{ .lo = 0x0FC6, .hi = 0x0FC6 },   .{ .lo = 0x102D, .hi = 0x1030 },
    .{ .lo = 0x1032, .hi = 0x1037 },   .{ .lo = 0x1039, .hi = 0x103A },   .{ .lo = 0x103D, .hi = 0x103E },   .{ .lo = 0x1058, .hi = 0x1059 },
    .{ .lo = 0x105E, .hi = 0x1060 },   .{ .lo = 0x1071, .hi = 0x1074 },   .{ .lo = 0x1082, .hi = 0x1082 },   .{ .lo = 0x1085, .hi = 0x1086 },
    .{ .lo = 0x108D, .hi = 0x108D },   .{ .lo = 0x109D, .hi = 0x109D },   .{ .lo = 0x135D, .hi = 0x135F },   .{ .lo = 0x1712, .hi = 0x1714 },
    .{ .lo = 0x1732, .hi = 0x1733 },   .{ .lo = 0x1752, .hi = 0x1753 },   .{ .lo = 0x1772, .hi = 0x1773 },   .{ .lo = 0x17B4, .hi = 0x17B5 },
    .{ .lo = 0x17B7, .hi = 0x17BD },   .{ .lo = 0x17C6, .hi = 0x17C6 },   .{ .lo = 0x17C9, .hi = 0x17D3 },   .{ .lo = 0x17DD, .hi = 0x17DD },
    .{ .lo = 0x180B, .hi = 0x180D },   .{ .lo = 0x180F, .hi = 0x180F },   .{ .lo = 0x1885, .hi = 0x1886 },   .{ .lo = 0x18A9, .hi = 0x18A9 },
    .{ .lo = 0x1920, .hi = 0x1922 },   .{ .lo = 0x1927, .hi = 0x1928 },   .{ .lo = 0x1932, .hi = 0x1932 },   .{ .lo = 0x1939, .hi = 0x193B },
    .{ .lo = 0x1A17, .hi = 0x1A18 },   .{ .lo = 0x1A1B, .hi = 0x1A1B },   .{ .lo = 0x1A56, .hi = 0x1A56 },   .{ .lo = 0x1A58, .hi = 0x1A5E },
    .{ .lo = 0x1A60, .hi = 0x1A60 },   .{ .lo = 0x1A62, .hi = 0x1A62 },   .{ .lo = 0x1A65, .hi = 0x1A6C },   .{ .lo = 0x1A73, .hi = 0x1A7C },
    .{ .lo = 0x1A7F, .hi = 0x1A7F },   .{ .lo = 0x1AB0, .hi = 0x1ABD },   .{ .lo = 0x1ABF, .hi = 0x1ACE },   .{ .lo = 0x1B00, .hi = 0x1B03 },
    .{ .lo = 0x1B34, .hi = 0x1B34 },   .{ .lo = 0x1B36, .hi = 0x1B3A },   .{ .lo = 0x1B3C, .hi = 0x1B3C },   .{ .lo = 0x1B42, .hi = 0x1B42 },
    .{ .lo = 0x1B6B, .hi = 0x1B73 },   .{ .lo = 0x1B80, .hi = 0x1B81 },   .{ .lo = 0x1BA2, .hi = 0x1BA5 },   .{ .lo = 0x1BA8, .hi = 0x1BA9 },
    .{ .lo = 0x1BAB, .hi = 0x1BAD },   .{ .lo = 0x1BE6, .hi = 0x1BE6 },   .{ .lo = 0x1BE8, .hi = 0x1BE9 },   .{ .lo = 0x1BED, .hi = 0x1BED },
    .{ .lo = 0x1BEF, .hi = 0x1BF1 },   .{ .lo = 0x1C2C, .hi = 0x1C33 },   .{ .lo = 0x1C36, .hi = 0x1C37 },   .{ .lo = 0x1CD0, .hi = 0x1CD2 },
    .{ .lo = 0x1CD4, .hi = 0x1CE0 },   .{ .lo = 0x1CE2, .hi = 0x1CE8 },   .{ .lo = 0x1CED, .hi = 0x1CED },   .{ .lo = 0x1CF4, .hi = 0x1CF4 },
    .{ .lo = 0x1CF8, .hi = 0x1CF9 },   .{ .lo = 0x1DC0, .hi = 0x1DFF },   .{ .lo = 0x20D0, .hi = 0x20DC },   .{ .lo = 0x20E1, .hi = 0x20E1 },
    .{ .lo = 0x20E5, .hi = 0x20F0 },   .{ .lo = 0x2CEF, .hi = 0x2CF1 },   .{ .lo = 0x2D7F, .hi = 0x2D7F },   .{ .lo = 0x2DE0, .hi = 0x2DFF },
    .{ .lo = 0x302A, .hi = 0x302D },   .{ .lo = 0x3099, .hi = 0x309A },   .{ .lo = 0xA66F, .hi = 0xA66F },   .{ .lo = 0xA674, .hi = 0xA67D },
    .{ .lo = 0xA69E, .hi = 0xA69F },   .{ .lo = 0xA6F0, .hi = 0xA6F1 },   .{ .lo = 0xA802, .hi = 0xA802 },   .{ .lo = 0xA806, .hi = 0xA806 },
    .{ .lo = 0xA80B, .hi = 0xA80B },   .{ .lo = 0xA825, .hi = 0xA826 },   .{ .lo = 0xA82C, .hi = 0xA82C },   .{ .lo = 0xA8C4, .hi = 0xA8C5 },
    .{ .lo = 0xA8E0, .hi = 0xA8F1 },   .{ .lo = 0xA8FF, .hi = 0xA8FF },   .{ .lo = 0xA926, .hi = 0xA92D },   .{ .lo = 0xA947, .hi = 0xA951 },
    .{ .lo = 0xA980, .hi = 0xA982 },   .{ .lo = 0xA9B3, .hi = 0xA9B3 },   .{ .lo = 0xA9B6, .hi = 0xA9B9 },   .{ .lo = 0xA9BC, .hi = 0xA9BD },
    .{ .lo = 0xA9E5, .hi = 0xA9E5 },   .{ .lo = 0xAA29, .hi = 0xAA2E },   .{ .lo = 0xAA31, .hi = 0xAA32 },   .{ .lo = 0xAA35, .hi = 0xAA36 },
    .{ .lo = 0xAA43, .hi = 0xAA43 },   .{ .lo = 0xAA4C, .hi = 0xAA4C },   .{ .lo = 0xAA7C, .hi = 0xAA7C },   .{ .lo = 0xAAB0, .hi = 0xAAB0 },
    .{ .lo = 0xAAB2, .hi = 0xAAB4 },   .{ .lo = 0xAAB7, .hi = 0xAAB8 },   .{ .lo = 0xAABE, .hi = 0xAABF },   .{ .lo = 0xAAC1, .hi = 0xAAC1 },
    .{ .lo = 0xAAEC, .hi = 0xAAED },   .{ .lo = 0xAAF6, .hi = 0xAAF6 },   .{ .lo = 0xABE5, .hi = 0xABE5 },   .{ .lo = 0xABE8, .hi = 0xABE8 },
    .{ .lo = 0xABED, .hi = 0xABED },   .{ .lo = 0xFB1E, .hi = 0xFB1E },   .{ .lo = 0xFE00, .hi = 0xFE0F },   .{ .lo = 0xFE20, .hi = 0xFE2F },
    .{ .lo = 0x101FD, .hi = 0x101FD }, .{ .lo = 0x102E0, .hi = 0x102E0 }, .{ .lo = 0x10376, .hi = 0x1037A }, .{ .lo = 0x10A01, .hi = 0x10A03 },
    .{ .lo = 0x10A05, .hi = 0x10A06 }, .{ .lo = 0x10A0C, .hi = 0x10A0F }, .{ .lo = 0x10A38, .hi = 0x10A3A }, .{ .lo = 0x10A3F, .hi = 0x10A3F },
    .{ .lo = 0x10AE5, .hi = 0x10AE6 }, .{ .lo = 0x10D24, .hi = 0x10D27 }, .{ .lo = 0x10D69, .hi = 0x10D6D }, .{ .lo = 0x10EAB, .hi = 0x10EAC },
    .{ .lo = 0x10EFC, .hi = 0x10EFF }, .{ .lo = 0x10F46, .hi = 0x10F50 }, .{ .lo = 0x10F82, .hi = 0x10F85 }, .{ .lo = 0x11001, .hi = 0x11001 },
    .{ .lo = 0x11038, .hi = 0x11046 }, .{ .lo = 0x11070, .hi = 0x11070 }, .{ .lo = 0x11073, .hi = 0x11074 }, .{ .lo = 0x1107F, .hi = 0x11081 },
    .{ .lo = 0x110B3, .hi = 0x110B6 }, .{ .lo = 0x110B9, .hi = 0x110BA }, .{ .lo = 0x110C2, .hi = 0x110C2 }, .{ .lo = 0x11100, .hi = 0x11102 },
    .{ .lo = 0x11127, .hi = 0x1112B }, .{ .lo = 0x1112D, .hi = 0x11134 }, .{ .lo = 0x11173, .hi = 0x11173 }, .{ .lo = 0x11180, .hi = 0x11181 },
    .{ .lo = 0x111B6, .hi = 0x111BE }, .{ .lo = 0x111C9, .hi = 0x111CC }, .{ .lo = 0x111CF, .hi = 0x111CF }, .{ .lo = 0x1122F, .hi = 0x11231 },
    .{ .lo = 0x11234, .hi = 0x11234 }, .{ .lo = 0x11236, .hi = 0x11237 }, .{ .lo = 0x1123E, .hi = 0x1123E }, .{ .lo = 0x11241, .hi = 0x11241 },
    .{ .lo = 0x112DF, .hi = 0x112DF }, .{ .lo = 0x112E3, .hi = 0x112EA }, .{ .lo = 0x11300, .hi = 0x11301 }, .{ .lo = 0x1133B, .hi = 0x1133C },
    .{ .lo = 0x11340, .hi = 0x11340 }, .{ .lo = 0x11366, .hi = 0x1136C }, .{ .lo = 0x11370, .hi = 0x11374 }, .{ .lo = 0x113BB, .hi = 0x113C0 },
    .{ .lo = 0x113CE, .hi = 0x113CE }, .{ .lo = 0x113D0, .hi = 0x113D0 }, .{ .lo = 0x113D2, .hi = 0x113D2 }, .{ .lo = 0x113E1, .hi = 0x113E2 },
    .{ .lo = 0x11438, .hi = 0x1143F }, .{ .lo = 0x11442, .hi = 0x11444 }, .{ .lo = 0x11446, .hi = 0x11446 }, .{ .lo = 0x1145E, .hi = 0x1145E },
    .{ .lo = 0x114B3, .hi = 0x114B8 }, .{ .lo = 0x114BA, .hi = 0x114BA }, .{ .lo = 0x114BF, .hi = 0x114C0 }, .{ .lo = 0x114C2, .hi = 0x114C3 },
    .{ .lo = 0x115B2, .hi = 0x115B5 }, .{ .lo = 0x115BC, .hi = 0x115BD }, .{ .lo = 0x115BF, .hi = 0x115C0 }, .{ .lo = 0x115DC, .hi = 0x115DD },
    .{ .lo = 0x11633, .hi = 0x1163A }, .{ .lo = 0x1163D, .hi = 0x1163D }, .{ .lo = 0x1163F, .hi = 0x11640 }, .{ .lo = 0x116AB, .hi = 0x116AB },
    .{ .lo = 0x116AD, .hi = 0x116AD }, .{ .lo = 0x116B0, .hi = 0x116B5 }, .{ .lo = 0x116B7, .hi = 0x116B7 }, .{ .lo = 0x1171D, .hi = 0x1171D },
    .{ .lo = 0x1171F, .hi = 0x1171F }, .{ .lo = 0x11722, .hi = 0x11725 }, .{ .lo = 0x11727, .hi = 0x1172B }, .{ .lo = 0x1182F, .hi = 0x11837 },
    .{ .lo = 0x11839, .hi = 0x1183A }, .{ .lo = 0x1193B, .hi = 0x1193C }, .{ .lo = 0x1193E, .hi = 0x1193E }, .{ .lo = 0x11943, .hi = 0x11943 },
    .{ .lo = 0x119D4, .hi = 0x119D7 }, .{ .lo = 0x119DA, .hi = 0x119DB }, .{ .lo = 0x119E0, .hi = 0x119E0 }, .{ .lo = 0x11A01, .hi = 0x11A0A },
    .{ .lo = 0x11A33, .hi = 0x11A38 }, .{ .lo = 0x11A3B, .hi = 0x11A3E }, .{ .lo = 0x11A47, .hi = 0x11A47 }, .{ .lo = 0x11A51, .hi = 0x11A56 },
    .{ .lo = 0x11A59, .hi = 0x11A5B }, .{ .lo = 0x11A8A, .hi = 0x11A96 }, .{ .lo = 0x11A98, .hi = 0x11A99 }, .{ .lo = 0x11C30, .hi = 0x11C36 },
    .{ .lo = 0x11C38, .hi = 0x11C3D }, .{ .lo = 0x11C3F, .hi = 0x11C3F }, .{ .lo = 0x11C92, .hi = 0x11CA7 }, .{ .lo = 0x11CAA, .hi = 0x11CB0 },
    .{ .lo = 0x11CB2, .hi = 0x11CB3 }, .{ .lo = 0x11CB5, .hi = 0x11CB6 }, .{ .lo = 0x11D31, .hi = 0x11D36 }, .{ .lo = 0x11D3A, .hi = 0x11D3A },
    .{ .lo = 0x11D3C, .hi = 0x11D3D }, .{ .lo = 0x11D3F, .hi = 0x11D45 }, .{ .lo = 0x11D47, .hi = 0x11D47 }, .{ .lo = 0x11D90, .hi = 0x11D91 },
    .{ .lo = 0x11D95, .hi = 0x11D95 }, .{ .lo = 0x11D97, .hi = 0x11D97 }, .{ .lo = 0x11EF3, .hi = 0x11EF4 }, .{ .lo = 0x11F00, .hi = 0x11F01 },
    .{ .lo = 0x11F36, .hi = 0x11F3A }, .{ .lo = 0x11F40, .hi = 0x11F40 }, .{ .lo = 0x11F42, .hi = 0x11F42 }, .{ .lo = 0x11F5A, .hi = 0x11F5A },
    .{ .lo = 0x13440, .hi = 0x13440 }, .{ .lo = 0x13447, .hi = 0x13455 }, .{ .lo = 0x1611E, .hi = 0x16129 }, .{ .lo = 0x1612D, .hi = 0x1612F },
    .{ .lo = 0x16AF0, .hi = 0x16AF4 }, .{ .lo = 0x16B30, .hi = 0x16B36 }, .{ .lo = 0x16F4F, .hi = 0x16F4F }, .{ .lo = 0x16F8F, .hi = 0x16F92 },
    .{ .lo = 0x16FE4, .hi = 0x16FE4 }, .{ .lo = 0x1BC9D, .hi = 0x1BC9E }, .{ .lo = 0x1CF00, .hi = 0x1CF2D }, .{ .lo = 0x1CF30, .hi = 0x1CF46 },
    .{ .lo = 0x1D167, .hi = 0x1D169 }, .{ .lo = 0x1D17B, .hi = 0x1D182 }, .{ .lo = 0x1D185, .hi = 0x1D18B }, .{ .lo = 0x1D1AA, .hi = 0x1D1AD },
    .{ .lo = 0x1D242, .hi = 0x1D244 }, .{ .lo = 0x1DA00, .hi = 0x1DA36 }, .{ .lo = 0x1DA3B, .hi = 0x1DA6C }, .{ .lo = 0x1DA75, .hi = 0x1DA75 },
    .{ .lo = 0x1DA84, .hi = 0x1DA84 }, .{ .lo = 0x1DA9B, .hi = 0x1DA9F }, .{ .lo = 0x1DAA1, .hi = 0x1DAAF }, .{ .lo = 0x1E000, .hi = 0x1E006 },
    .{ .lo = 0x1E008, .hi = 0x1E018 }, .{ .lo = 0x1E01B, .hi = 0x1E021 }, .{ .lo = 0x1E023, .hi = 0x1E024 }, .{ .lo = 0x1E026, .hi = 0x1E02A },
    .{ .lo = 0x1E08F, .hi = 0x1E08F }, .{ .lo = 0x1E130, .hi = 0x1E136 }, .{ .lo = 0x1E2AE, .hi = 0x1E2AE }, .{ .lo = 0x1E2EC, .hi = 0x1E2EF },
    .{ .lo = 0x1E4EC, .hi = 0x1E4EF }, .{ .lo = 0x1E5EE, .hi = 0x1E5EF }, .{ .lo = 0x1E8D0, .hi = 0x1E8D6 }, .{ .lo = 0x1E944, .hi = 0x1E94A },
    .{ .lo = 0xE0100, .hi = 0xE01EF },
};

const cat_Mc = [_]Range{
    .{ .lo = 0x0903, .hi = 0x0903 },   .{ .lo = 0x093B, .hi = 0x093B },   .{ .lo = 0x093E, .hi = 0x0940 },   .{ .lo = 0x0949, .hi = 0x094C },
    .{ .lo = 0x094E, .hi = 0x094F },   .{ .lo = 0x0982, .hi = 0x0983 },   .{ .lo = 0x09BE, .hi = 0x09C0 },   .{ .lo = 0x09C7, .hi = 0x09C8 },
    .{ .lo = 0x09CB, .hi = 0x09CC },   .{ .lo = 0x09D7, .hi = 0x09D7 },   .{ .lo = 0x0A03, .hi = 0x0A03 },   .{ .lo = 0x0A3E, .hi = 0x0A40 },
    .{ .lo = 0x0A83, .hi = 0x0A83 },   .{ .lo = 0x0ABE, .hi = 0x0AC0 },   .{ .lo = 0x0AC9, .hi = 0x0AC9 },   .{ .lo = 0x0ACB, .hi = 0x0ACC },
    .{ .lo = 0x0B02, .hi = 0x0B03 },   .{ .lo = 0x0B3E, .hi = 0x0B3E },   .{ .lo = 0x0B40, .hi = 0x0B40 },   .{ .lo = 0x0B47, .hi = 0x0B48 },
    .{ .lo = 0x0B4B, .hi = 0x0B4C },   .{ .lo = 0x0B57, .hi = 0x0B57 },   .{ .lo = 0x0BBE, .hi = 0x0BBF },   .{ .lo = 0x0BC1, .hi = 0x0BC2 },
    .{ .lo = 0x0BC6, .hi = 0x0BC8 },   .{ .lo = 0x0BCA, .hi = 0x0BCC },   .{ .lo = 0x0BD7, .hi = 0x0BD7 },   .{ .lo = 0x0C01, .hi = 0x0C03 },
    .{ .lo = 0x0C41, .hi = 0x0C44 },   .{ .lo = 0x0C82, .hi = 0x0C83 },   .{ .lo = 0x0CBE, .hi = 0x0CBE },   .{ .lo = 0x0CC0, .hi = 0x0CC4 },
    .{ .lo = 0x0CC7, .hi = 0x0CC8 },   .{ .lo = 0x0CCA, .hi = 0x0CCB },   .{ .lo = 0x0CD5, .hi = 0x0CD6 },   .{ .lo = 0x0CF3, .hi = 0x0CF3 },
    .{ .lo = 0x0D02, .hi = 0x0D03 },   .{ .lo = 0x0D3E, .hi = 0x0D40 },   .{ .lo = 0x0D46, .hi = 0x0D48 },   .{ .lo = 0x0D4A, .hi = 0x0D4C },
    .{ .lo = 0x0D57, .hi = 0x0D57 },   .{ .lo = 0x0D82, .hi = 0x0D83 },   .{ .lo = 0x0DCF, .hi = 0x0DD1 },   .{ .lo = 0x0DD8, .hi = 0x0DDF },
    .{ .lo = 0x0DF2, .hi = 0x0DF3 },   .{ .lo = 0x0F3E, .hi = 0x0F3F },   .{ .lo = 0x0F7F, .hi = 0x0F7F },   .{ .lo = 0x102B, .hi = 0x102C },
    .{ .lo = 0x1031, .hi = 0x1031 },   .{ .lo = 0x1038, .hi = 0x1038 },   .{ .lo = 0x103B, .hi = 0x103C },   .{ .lo = 0x1056, .hi = 0x1057 },
    .{ .lo = 0x1062, .hi = 0x1064 },   .{ .lo = 0x1067, .hi = 0x106D },   .{ .lo = 0x1083, .hi = 0x1084 },   .{ .lo = 0x1087, .hi = 0x108C },
    .{ .lo = 0x108F, .hi = 0x108F },   .{ .lo = 0x109A, .hi = 0x109C },   .{ .lo = 0x1715, .hi = 0x1715 },   .{ .lo = 0x1734, .hi = 0x1734 },
    .{ .lo = 0x17B6, .hi = 0x17B6 },   .{ .lo = 0x17BE, .hi = 0x17C5 },   .{ .lo = 0x17C7, .hi = 0x17C8 },   .{ .lo = 0x1923, .hi = 0x1926 },
    .{ .lo = 0x1929, .hi = 0x192B },   .{ .lo = 0x1930, .hi = 0x1931 },   .{ .lo = 0x1933, .hi = 0x1938 },   .{ .lo = 0x1A19, .hi = 0x1A1A },
    .{ .lo = 0x1A55, .hi = 0x1A55 },   .{ .lo = 0x1A57, .hi = 0x1A57 },   .{ .lo = 0x1A61, .hi = 0x1A61 },   .{ .lo = 0x1A63, .hi = 0x1A64 },
    .{ .lo = 0x1A6D, .hi = 0x1A72 },   .{ .lo = 0x1B04, .hi = 0x1B04 },   .{ .lo = 0x1B35, .hi = 0x1B35 },   .{ .lo = 0x1B3B, .hi = 0x1B3B },
    .{ .lo = 0x1B3D, .hi = 0x1B41 },   .{ .lo = 0x1B43, .hi = 0x1B44 },   .{ .lo = 0x1B82, .hi = 0x1B82 },   .{ .lo = 0x1BA1, .hi = 0x1BA1 },
    .{ .lo = 0x1BA6, .hi = 0x1BA7 },   .{ .lo = 0x1BAA, .hi = 0x1BAA },   .{ .lo = 0x1BE7, .hi = 0x1BE7 },   .{ .lo = 0x1BEA, .hi = 0x1BEC },
    .{ .lo = 0x1BEE, .hi = 0x1BEE },   .{ .lo = 0x1BF2, .hi = 0x1BF3 },   .{ .lo = 0x1C24, .hi = 0x1C2B },   .{ .lo = 0x1C34, .hi = 0x1C35 },
    .{ .lo = 0x1CE1, .hi = 0x1CE1 },   .{ .lo = 0x1CF7, .hi = 0x1CF7 },   .{ .lo = 0x302E, .hi = 0x302F },   .{ .lo = 0xA823, .hi = 0xA824 },
    .{ .lo = 0xA827, .hi = 0xA827 },   .{ .lo = 0xA880, .hi = 0xA881 },   .{ .lo = 0xA8B4, .hi = 0xA8C3 },   .{ .lo = 0xA952, .hi = 0xA953 },
    .{ .lo = 0xA983, .hi = 0xA983 },   .{ .lo = 0xA9B4, .hi = 0xA9B5 },   .{ .lo = 0xA9BA, .hi = 0xA9BB },   .{ .lo = 0xA9BE, .hi = 0xA9C0 },
    .{ .lo = 0xAA2F, .hi = 0xAA30 },   .{ .lo = 0xAA33, .hi = 0xAA34 },   .{ .lo = 0xAA4D, .hi = 0xAA4D },   .{ .lo = 0xAA7B, .hi = 0xAA7B },
    .{ .lo = 0xAA7D, .hi = 0xAA7D },   .{ .lo = 0xAAEB, .hi = 0xAAEB },   .{ .lo = 0xAAEE, .hi = 0xAAEF },   .{ .lo = 0xAAF5, .hi = 0xAAF5 },
    .{ .lo = 0xABE3, .hi = 0xABE4 },   .{ .lo = 0xABE6, .hi = 0xABE7 },   .{ .lo = 0xABE9, .hi = 0xABEA },   .{ .lo = 0xABEC, .hi = 0xABEC },
    .{ .lo = 0x11000, .hi = 0x11000 }, .{ .lo = 0x11002, .hi = 0x11002 }, .{ .lo = 0x11082, .hi = 0x11082 }, .{ .lo = 0x110B0, .hi = 0x110B2 },
    .{ .lo = 0x110B7, .hi = 0x110B8 }, .{ .lo = 0x1112C, .hi = 0x1112C }, .{ .lo = 0x11145, .hi = 0x11146 }, .{ .lo = 0x11182, .hi = 0x11182 },
    .{ .lo = 0x111B3, .hi = 0x111B5 }, .{ .lo = 0x111BF, .hi = 0x111C0 }, .{ .lo = 0x111CE, .hi = 0x111CE }, .{ .lo = 0x1122C, .hi = 0x1122E },
    .{ .lo = 0x11232, .hi = 0x11233 }, .{ .lo = 0x11235, .hi = 0x11235 }, .{ .lo = 0x112E0, .hi = 0x112E2 }, .{ .lo = 0x11302, .hi = 0x11303 },
    .{ .lo = 0x1133E, .hi = 0x1133F }, .{ .lo = 0x11341, .hi = 0x11344 }, .{ .lo = 0x11347, .hi = 0x11348 }, .{ .lo = 0x1134B, .hi = 0x1134D },
    .{ .lo = 0x11357, .hi = 0x11357 }, .{ .lo = 0x11362, .hi = 0x11363 }, .{ .lo = 0x113B8, .hi = 0x113BA }, .{ .lo = 0x113C2, .hi = 0x113C2 },
    .{ .lo = 0x113C5, .hi = 0x113C5 }, .{ .lo = 0x113C7, .hi = 0x113CA }, .{ .lo = 0x113CC, .hi = 0x113CD }, .{ .lo = 0x113CF, .hi = 0x113CF },
    .{ .lo = 0x11435, .hi = 0x11437 }, .{ .lo = 0x11440, .hi = 0x11441 }, .{ .lo = 0x11445, .hi = 0x11445 }, .{ .lo = 0x114B0, .hi = 0x114B2 },
    .{ .lo = 0x114B9, .hi = 0x114B9 }, .{ .lo = 0x114BB, .hi = 0x114BE }, .{ .lo = 0x114C1, .hi = 0x114C1 }, .{ .lo = 0x115AF, .hi = 0x115B1 },
    .{ .lo = 0x115B8, .hi = 0x115BB }, .{ .lo = 0x115BE, .hi = 0x115BE }, .{ .lo = 0x11630, .hi = 0x11632 }, .{ .lo = 0x1163B, .hi = 0x1163C },
    .{ .lo = 0x1163E, .hi = 0x1163E }, .{ .lo = 0x116AC, .hi = 0x116AC }, .{ .lo = 0x116AE, .hi = 0x116AF }, .{ .lo = 0x116B6, .hi = 0x116B6 },
    .{ .lo = 0x1171E, .hi = 0x1171E }, .{ .lo = 0x11720, .hi = 0x11721 }, .{ .lo = 0x11726, .hi = 0x11726 }, .{ .lo = 0x1182C, .hi = 0x1182E },
    .{ .lo = 0x11838, .hi = 0x11838 }, .{ .lo = 0x11930, .hi = 0x11935 }, .{ .lo = 0x11937, .hi = 0x11938 }, .{ .lo = 0x1193D, .hi = 0x1193D },
    .{ .lo = 0x11940, .hi = 0x11940 }, .{ .lo = 0x11942, .hi = 0x11942 }, .{ .lo = 0x119D1, .hi = 0x119D3 }, .{ .lo = 0x119DC, .hi = 0x119DF },
    .{ .lo = 0x119E4, .hi = 0x119E4 }, .{ .lo = 0x11A39, .hi = 0x11A39 }, .{ .lo = 0x11A57, .hi = 0x11A58 }, .{ .lo = 0x11A97, .hi = 0x11A97 },
    .{ .lo = 0x11C2F, .hi = 0x11C2F }, .{ .lo = 0x11C3E, .hi = 0x11C3E }, .{ .lo = 0x11CA9, .hi = 0x11CA9 }, .{ .lo = 0x11CB1, .hi = 0x11CB1 },
    .{ .lo = 0x11CB4, .hi = 0x11CB4 }, .{ .lo = 0x11D8A, .hi = 0x11D8E }, .{ .lo = 0x11D93, .hi = 0x11D94 }, .{ .lo = 0x11D96, .hi = 0x11D96 },
    .{ .lo = 0x11EF5, .hi = 0x11EF6 }, .{ .lo = 0x11F03, .hi = 0x11F03 }, .{ .lo = 0x11F34, .hi = 0x11F35 }, .{ .lo = 0x11F3E, .hi = 0x11F3F },
    .{ .lo = 0x11F41, .hi = 0x11F41 }, .{ .lo = 0x1612A, .hi = 0x1612C }, .{ .lo = 0x16F51, .hi = 0x16F87 }, .{ .lo = 0x16FF0, .hi = 0x16FF1 },
    .{ .lo = 0x1D165, .hi = 0x1D166 }, .{ .lo = 0x1D16D, .hi = 0x1D172 },
};

const cat_Me = [_]Range{
    .{ .lo = 0x0488, .hi = 0x0489 }, .{ .lo = 0x1ABE, .hi = 0x1ABE }, .{ .lo = 0x20DD, .hi = 0x20E0 }, .{ .lo = 0x20E2, .hi = 0x20E4 },
    .{ .lo = 0xA670, .hi = 0xA672 },
};

const cat_Nd = [_]Range{
    .{ .lo = 0x0030, .hi = 0x0039 },   .{ .lo = 0x0660, .hi = 0x0669 },   .{ .lo = 0x06F0, .hi = 0x06F9 },   .{ .lo = 0x07C0, .hi = 0x07C9 },
    .{ .lo = 0x0966, .hi = 0x096F },   .{ .lo = 0x09E6, .hi = 0x09EF },   .{ .lo = 0x0A66, .hi = 0x0A6F },   .{ .lo = 0x0AE6, .hi = 0x0AEF },
    .{ .lo = 0x0B66, .hi = 0x0B6F },   .{ .lo = 0x0BE6, .hi = 0x0BEF },   .{ .lo = 0x0C66, .hi = 0x0C6F },   .{ .lo = 0x0CE6, .hi = 0x0CEF },
    .{ .lo = 0x0D66, .hi = 0x0D6F },   .{ .lo = 0x0DE6, .hi = 0x0DEF },   .{ .lo = 0x0E50, .hi = 0x0E59 },   .{ .lo = 0x0ED0, .hi = 0x0ED9 },
    .{ .lo = 0x0F20, .hi = 0x0F29 },   .{ .lo = 0x1040, .hi = 0x1049 },   .{ .lo = 0x1090, .hi = 0x1099 },   .{ .lo = 0x17E0, .hi = 0x17E9 },
    .{ .lo = 0x1810, .hi = 0x1819 },   .{ .lo = 0x1946, .hi = 0x194F },   .{ .lo = 0x19D0, .hi = 0x19D9 },   .{ .lo = 0x1A80, .hi = 0x1A89 },
    .{ .lo = 0x1A90, .hi = 0x1A99 },   .{ .lo = 0x1B50, .hi = 0x1B59 },   .{ .lo = 0x1BB0, .hi = 0x1BB9 },   .{ .lo = 0x1C40, .hi = 0x1C49 },
    .{ .lo = 0x1C50, .hi = 0x1C59 },   .{ .lo = 0xA620, .hi = 0xA629 },   .{ .lo = 0xA8D0, .hi = 0xA8D9 },   .{ .lo = 0xA900, .hi = 0xA909 },
    .{ .lo = 0xA9D0, .hi = 0xA9D9 },   .{ .lo = 0xA9F0, .hi = 0xA9F9 },   .{ .lo = 0xAA50, .hi = 0xAA59 },   .{ .lo = 0xABF0, .hi = 0xABF9 },
    .{ .lo = 0xFF10, .hi = 0xFF19 },   .{ .lo = 0x104A0, .hi = 0x104A9 }, .{ .lo = 0x10D30, .hi = 0x10D39 }, .{ .lo = 0x10D40, .hi = 0x10D49 },
    .{ .lo = 0x11066, .hi = 0x1106F }, .{ .lo = 0x110F0, .hi = 0x110F9 }, .{ .lo = 0x11136, .hi = 0x1113F }, .{ .lo = 0x111D0, .hi = 0x111D9 },
    .{ .lo = 0x112F0, .hi = 0x112F9 }, .{ .lo = 0x11450, .hi = 0x11459 }, .{ .lo = 0x114D0, .hi = 0x114D9 }, .{ .lo = 0x11650, .hi = 0x11659 },
    .{ .lo = 0x116C0, .hi = 0x116C9 }, .{ .lo = 0x116D0, .hi = 0x116E3 }, .{ .lo = 0x11730, .hi = 0x11739 }, .{ .lo = 0x118E0, .hi = 0x118E9 },
    .{ .lo = 0x11950, .hi = 0x11959 }, .{ .lo = 0x11BF0, .hi = 0x11BF9 }, .{ .lo = 0x11C50, .hi = 0x11C59 }, .{ .lo = 0x11D50, .hi = 0x11D59 },
    .{ .lo = 0x11DA0, .hi = 0x11DA9 }, .{ .lo = 0x11F50, .hi = 0x11F59 }, .{ .lo = 0x16130, .hi = 0x16139 }, .{ .lo = 0x16A60, .hi = 0x16A69 },
    .{ .lo = 0x16AC0, .hi = 0x16AC9 }, .{ .lo = 0x16B50, .hi = 0x16B59 }, .{ .lo = 0x16D70, .hi = 0x16D79 }, .{ .lo = 0x1CCF0, .hi = 0x1CCF9 },
    .{ .lo = 0x1D7CE, .hi = 0x1D7FF }, .{ .lo = 0x1E140, .hi = 0x1E149 }, .{ .lo = 0x1E2F0, .hi = 0x1E2F9 }, .{ .lo = 0x1E4F0, .hi = 0x1E4F9 },
    .{ .lo = 0x1E5F1, .hi = 0x1E5FA }, .{ .lo = 0x1E950, .hi = 0x1E959 }, .{ .lo = 0x1FBF0, .hi = 0x1FBF9 },
};

const cat_Nl = [_]Range{
    .{ .lo = 0x16EE, .hi = 0x16F0 },   .{ .lo = 0x2160, .hi = 0x2182 },   .{ .lo = 0x2185, .hi = 0x2188 },   .{ .lo = 0x3007, .hi = 0x3007 },
    .{ .lo = 0x3021, .hi = 0x3029 },   .{ .lo = 0x3038, .hi = 0x303A },   .{ .lo = 0xA6E6, .hi = 0xA6EF },   .{ .lo = 0x10140, .hi = 0x10174 },
    .{ .lo = 0x10341, .hi = 0x10341 }, .{ .lo = 0x1034A, .hi = 0x1034A }, .{ .lo = 0x103D1, .hi = 0x103D5 }, .{ .lo = 0x12400, .hi = 0x1246E },
};

// General_Category = Cf (Format) — Unicode 16.0.
// Used by the Bert `clean_text` pass: HF's `is_control` matches
// `c.is_other()` which spans Cc | Cf | Cn | Co (Unicode TR44 Table 12).
// Cc is already handled with explicit ASCII C0/C1 ranges in
// `isBertControlCp` (cheaper than a table lookup); Cn and Co are
// effectively unreachable in real text and not modeled here. Cf is the
// category that contains the bidi controls (LRM/RLM, LRE/RLE/PDF,
// LRI/RLI/FSI/PDI), zero-width joiners, soft hyphen, BOM, and the
// language tag block (U+E0001, U+E0020-U+E007F) — all of which HF
// strips and ztok was previously leaving in. Without this we lost
// ~80/1000 lines on bench/corpora/unicode_stress.txt to bidi-isolated
// Arabic/Hebrew snippets and another 16 to TAG-emoji flag sequences.
const cat_Cf = [_]Range{
    .{ .lo = 0x00AD, .hi = 0x00AD },   .{ .lo = 0x0600, .hi = 0x0605 },   .{ .lo = 0x061C, .hi = 0x061C },   .{ .lo = 0x06DD, .hi = 0x06DD },
    .{ .lo = 0x070F, .hi = 0x070F },   .{ .lo = 0x0890, .hi = 0x0891 },   .{ .lo = 0x08E2, .hi = 0x08E2 },   .{ .lo = 0x180E, .hi = 0x180E },
    .{ .lo = 0x200B, .hi = 0x200F },   .{ .lo = 0x202A, .hi = 0x202E },   .{ .lo = 0x2060, .hi = 0x2064 },   .{ .lo = 0x2066, .hi = 0x206F },
    .{ .lo = 0xFEFF, .hi = 0xFEFF },   .{ .lo = 0xFFF9, .hi = 0xFFFB },   .{ .lo = 0x110BD, .hi = 0x110BD }, .{ .lo = 0x110CD, .hi = 0x110CD },
    .{ .lo = 0x13430, .hi = 0x1343F }, .{ .lo = 0x1BCA0, .hi = 0x1BCA3 }, .{ .lo = 0x1D173, .hi = 0x1D17A }, .{ .lo = 0xE0001, .hi = 0xE0001 },
    .{ .lo = 0xE0020, .hi = 0xE007F },
};

// General_Category = P (union of Pc | Pd | Pe | Pf | Pi | Po | Ps) — Unicode 16.0.
// Used by the HF `Punctuation` pre-tokenizer to decide span boundaries.
// HF's `is_punc` is `char::is_ascii_punctuation(&x) || x.is_punctuation()`
// where `is_punctuation()` comes from the `unicode_categories` crate and
// returns true exactly for Pc/Pd/Pe/Pf/Pi/Po/Ps. Falcon-7B's pre_tokenizer
// chain starts with `Punctuation(Contiguous)`, so any over-broad approximation
// (e.g. "the whole 0x2000-0x206F block is punct") splits format characters
// like U+2068/U+2069 (FIRST STRONG ISOLATE / POP DIRECTIONAL ISOLATE) out of
// the leading-space run and corrupts the subsequent ByteLevel BPE merges.
// Without this we lost 80/1000 lines on bench/corpora/unicode_stress.txt to
// bidi-isolated Arabic/Hebrew snippets and 1/10000 lines on code.txt.
const cat_P = [_]Range{
    .{ .lo = 0x0021, .hi = 0x0023 },   .{ .lo = 0x0025, .hi = 0x002A },   .{ .lo = 0x002C, .hi = 0x002F },   .{ .lo = 0x003A, .hi = 0x003B },
    .{ .lo = 0x003F, .hi = 0x0040 },   .{ .lo = 0x005B, .hi = 0x005D },   .{ .lo = 0x005F, .hi = 0x005F },   .{ .lo = 0x007B, .hi = 0x007B },
    .{ .lo = 0x007D, .hi = 0x007D },   .{ .lo = 0x00A1, .hi = 0x00A1 },   .{ .lo = 0x00A7, .hi = 0x00A7 },   .{ .lo = 0x00AB, .hi = 0x00AB },
    .{ .lo = 0x00B6, .hi = 0x00B7 },   .{ .lo = 0x00BB, .hi = 0x00BB },   .{ .lo = 0x00BF, .hi = 0x00BF },   .{ .lo = 0x037E, .hi = 0x037E },
    .{ .lo = 0x0387, .hi = 0x0387 },   .{ .lo = 0x055A, .hi = 0x055F },   .{ .lo = 0x0589, .hi = 0x058A },   .{ .lo = 0x05BE, .hi = 0x05BE },
    .{ .lo = 0x05C0, .hi = 0x05C0 },   .{ .lo = 0x05C3, .hi = 0x05C3 },   .{ .lo = 0x05C6, .hi = 0x05C6 },   .{ .lo = 0x05F3, .hi = 0x05F4 },
    .{ .lo = 0x0609, .hi = 0x060A },   .{ .lo = 0x060C, .hi = 0x060D },   .{ .lo = 0x061B, .hi = 0x061B },   .{ .lo = 0x061D, .hi = 0x061F },
    .{ .lo = 0x066A, .hi = 0x066D },   .{ .lo = 0x06D4, .hi = 0x06D4 },   .{ .lo = 0x0700, .hi = 0x070D },   .{ .lo = 0x07F7, .hi = 0x07F9 },
    .{ .lo = 0x0830, .hi = 0x083E },   .{ .lo = 0x085E, .hi = 0x085E },   .{ .lo = 0x0964, .hi = 0x0965 },   .{ .lo = 0x0970, .hi = 0x0970 },
    .{ .lo = 0x09FD, .hi = 0x09FD },   .{ .lo = 0x0A76, .hi = 0x0A76 },   .{ .lo = 0x0AF0, .hi = 0x0AF0 },   .{ .lo = 0x0C77, .hi = 0x0C77 },
    .{ .lo = 0x0C84, .hi = 0x0C84 },   .{ .lo = 0x0DF4, .hi = 0x0DF4 },   .{ .lo = 0x0E4F, .hi = 0x0E4F },   .{ .lo = 0x0E5A, .hi = 0x0E5B },
    .{ .lo = 0x0F04, .hi = 0x0F12 },   .{ .lo = 0x0F14, .hi = 0x0F14 },   .{ .lo = 0x0F3A, .hi = 0x0F3D },   .{ .lo = 0x0F85, .hi = 0x0F85 },
    .{ .lo = 0x0FD0, .hi = 0x0FD4 },   .{ .lo = 0x0FD9, .hi = 0x0FDA },   .{ .lo = 0x104A, .hi = 0x104F },   .{ .lo = 0x10FB, .hi = 0x10FB },
    .{ .lo = 0x1360, .hi = 0x1368 },   .{ .lo = 0x1400, .hi = 0x1400 },   .{ .lo = 0x166E, .hi = 0x166E },   .{ .lo = 0x169B, .hi = 0x169C },
    .{ .lo = 0x16EB, .hi = 0x16ED },   .{ .lo = 0x1735, .hi = 0x1736 },   .{ .lo = 0x17D4, .hi = 0x17D6 },   .{ .lo = 0x17D8, .hi = 0x17DA },
    .{ .lo = 0x1800, .hi = 0x180A },   .{ .lo = 0x1944, .hi = 0x1945 },   .{ .lo = 0x1A1E, .hi = 0x1A1F },   .{ .lo = 0x1AA0, .hi = 0x1AA6 },
    .{ .lo = 0x1AA8, .hi = 0x1AAD },   .{ .lo = 0x1B4E, .hi = 0x1B4F },   .{ .lo = 0x1B5A, .hi = 0x1B60 },   .{ .lo = 0x1B7D, .hi = 0x1B7F },
    .{ .lo = 0x1BFC, .hi = 0x1BFF },   .{ .lo = 0x1C3B, .hi = 0x1C3F },   .{ .lo = 0x1C7E, .hi = 0x1C7F },   .{ .lo = 0x1CC0, .hi = 0x1CC7 },
    .{ .lo = 0x1CD3, .hi = 0x1CD3 },   .{ .lo = 0x2010, .hi = 0x2027 },   .{ .lo = 0x2030, .hi = 0x2043 },   .{ .lo = 0x2045, .hi = 0x2051 },
    .{ .lo = 0x2053, .hi = 0x205E },   .{ .lo = 0x207D, .hi = 0x207E },   .{ .lo = 0x208D, .hi = 0x208E },   .{ .lo = 0x2308, .hi = 0x230B },
    .{ .lo = 0x2329, .hi = 0x232A },   .{ .lo = 0x2768, .hi = 0x2775 },   .{ .lo = 0x27C5, .hi = 0x27C6 },   .{ .lo = 0x27E6, .hi = 0x27EF },
    .{ .lo = 0x2983, .hi = 0x2998 },   .{ .lo = 0x29D8, .hi = 0x29DB },   .{ .lo = 0x29FC, .hi = 0x29FD },   .{ .lo = 0x2CF9, .hi = 0x2CFC },
    .{ .lo = 0x2CFE, .hi = 0x2CFF },   .{ .lo = 0x2D70, .hi = 0x2D70 },   .{ .lo = 0x2E00, .hi = 0x2E2E },   .{ .lo = 0x2E30, .hi = 0x2E4F },
    .{ .lo = 0x2E52, .hi = 0x2E5D },   .{ .lo = 0x3001, .hi = 0x3003 },   .{ .lo = 0x3008, .hi = 0x3011 },   .{ .lo = 0x3014, .hi = 0x301F },
    .{ .lo = 0x3030, .hi = 0x3030 },   .{ .lo = 0x303D, .hi = 0x303D },   .{ .lo = 0x30A0, .hi = 0x30A0 },   .{ .lo = 0x30FB, .hi = 0x30FB },
    .{ .lo = 0xA4FE, .hi = 0xA4FF },   .{ .lo = 0xA60D, .hi = 0xA60F },   .{ .lo = 0xA673, .hi = 0xA673 },   .{ .lo = 0xA67E, .hi = 0xA67E },
    .{ .lo = 0xA6F2, .hi = 0xA6F7 },   .{ .lo = 0xA874, .hi = 0xA877 },   .{ .lo = 0xA8CE, .hi = 0xA8CF },   .{ .lo = 0xA8F8, .hi = 0xA8FA },
    .{ .lo = 0xA8FC, .hi = 0xA8FC },   .{ .lo = 0xA92E, .hi = 0xA92F },   .{ .lo = 0xA95F, .hi = 0xA95F },   .{ .lo = 0xA9C1, .hi = 0xA9CD },
    .{ .lo = 0xA9DE, .hi = 0xA9DF },   .{ .lo = 0xAA5C, .hi = 0xAA5F },   .{ .lo = 0xAADE, .hi = 0xAADF },   .{ .lo = 0xAAF0, .hi = 0xAAF1 },
    .{ .lo = 0xABEB, .hi = 0xABEB },   .{ .lo = 0xFD3E, .hi = 0xFD3F },   .{ .lo = 0xFE10, .hi = 0xFE19 },   .{ .lo = 0xFE30, .hi = 0xFE52 },
    .{ .lo = 0xFE54, .hi = 0xFE61 },   .{ .lo = 0xFE63, .hi = 0xFE63 },   .{ .lo = 0xFE68, .hi = 0xFE68 },   .{ .lo = 0xFE6A, .hi = 0xFE6B },
    .{ .lo = 0xFF01, .hi = 0xFF03 },   .{ .lo = 0xFF05, .hi = 0xFF0A },   .{ .lo = 0xFF0C, .hi = 0xFF0F },   .{ .lo = 0xFF1A, .hi = 0xFF1B },
    .{ .lo = 0xFF1F, .hi = 0xFF20 },   .{ .lo = 0xFF3B, .hi = 0xFF3D },   .{ .lo = 0xFF3F, .hi = 0xFF3F },   .{ .lo = 0xFF5B, .hi = 0xFF5B },
    .{ .lo = 0xFF5D, .hi = 0xFF5D },   .{ .lo = 0xFF5F, .hi = 0xFF65 },   .{ .lo = 0x10100, .hi = 0x10102 }, .{ .lo = 0x1039F, .hi = 0x1039F },
    .{ .lo = 0x103D0, .hi = 0x103D0 }, .{ .lo = 0x1056F, .hi = 0x1056F }, .{ .lo = 0x10857, .hi = 0x10857 }, .{ .lo = 0x1091F, .hi = 0x1091F },
    .{ .lo = 0x1093F, .hi = 0x1093F }, .{ .lo = 0x10A50, .hi = 0x10A58 }, .{ .lo = 0x10A7F, .hi = 0x10A7F }, .{ .lo = 0x10AF0, .hi = 0x10AF6 },
    .{ .lo = 0x10B39, .hi = 0x10B3F }, .{ .lo = 0x10B99, .hi = 0x10B9C }, .{ .lo = 0x10D6E, .hi = 0x10D6E }, .{ .lo = 0x10EAD, .hi = 0x10EAD },
    .{ .lo = 0x10F55, .hi = 0x10F59 }, .{ .lo = 0x10F86, .hi = 0x10F89 }, .{ .lo = 0x11047, .hi = 0x1104D }, .{ .lo = 0x110BB, .hi = 0x110BC },
    .{ .lo = 0x110BE, .hi = 0x110C1 }, .{ .lo = 0x11140, .hi = 0x11143 }, .{ .lo = 0x11174, .hi = 0x11175 }, .{ .lo = 0x111C5, .hi = 0x111C8 },
    .{ .lo = 0x111CD, .hi = 0x111CD }, .{ .lo = 0x111DB, .hi = 0x111DB }, .{ .lo = 0x111DD, .hi = 0x111DF }, .{ .lo = 0x11238, .hi = 0x1123D },
    .{ .lo = 0x112A9, .hi = 0x112A9 }, .{ .lo = 0x113D4, .hi = 0x113D5 }, .{ .lo = 0x113D7, .hi = 0x113D8 }, .{ .lo = 0x1144B, .hi = 0x1144F },
    .{ .lo = 0x1145A, .hi = 0x1145B }, .{ .lo = 0x1145D, .hi = 0x1145D }, .{ .lo = 0x114C6, .hi = 0x114C6 }, .{ .lo = 0x115C1, .hi = 0x115D7 },
    .{ .lo = 0x11641, .hi = 0x11643 }, .{ .lo = 0x11660, .hi = 0x1166C }, .{ .lo = 0x116B9, .hi = 0x116B9 }, .{ .lo = 0x1173C, .hi = 0x1173E },
    .{ .lo = 0x1183B, .hi = 0x1183B }, .{ .lo = 0x11944, .hi = 0x11946 }, .{ .lo = 0x119E2, .hi = 0x119E2 }, .{ .lo = 0x11A3F, .hi = 0x11A46 },
    .{ .lo = 0x11A9A, .hi = 0x11A9C }, .{ .lo = 0x11A9E, .hi = 0x11AA2 }, .{ .lo = 0x11B00, .hi = 0x11B09 }, .{ .lo = 0x11BE1, .hi = 0x11BE1 },
    .{ .lo = 0x11C41, .hi = 0x11C45 }, .{ .lo = 0x11C70, .hi = 0x11C71 }, .{ .lo = 0x11EF7, .hi = 0x11EF8 }, .{ .lo = 0x11F43, .hi = 0x11F4F },
    .{ .lo = 0x11FFF, .hi = 0x11FFF }, .{ .lo = 0x12470, .hi = 0x12474 }, .{ .lo = 0x12FF1, .hi = 0x12FF2 }, .{ .lo = 0x16A6E, .hi = 0x16A6F },
    .{ .lo = 0x16AF5, .hi = 0x16AF5 }, .{ .lo = 0x16B37, .hi = 0x16B3B }, .{ .lo = 0x16B44, .hi = 0x16B44 }, .{ .lo = 0x16D6D, .hi = 0x16D6F },
    .{ .lo = 0x16E97, .hi = 0x16E9A }, .{ .lo = 0x16FE2, .hi = 0x16FE2 }, .{ .lo = 0x1BC9F, .hi = 0x1BC9F }, .{ .lo = 0x1DA87, .hi = 0x1DA8B },
    .{ .lo = 0x1E5FF, .hi = 0x1E5FF }, .{ .lo = 0x1E95E, .hi = 0x1E95F },
};

const cat_No = [_]Range{
    .{ .lo = 0x00B2, .hi = 0x00B3 },   .{ .lo = 0x00B9, .hi = 0x00B9 },   .{ .lo = 0x00BC, .hi = 0x00BE },   .{ .lo = 0x09F4, .hi = 0x09F9 },
    .{ .lo = 0x0B72, .hi = 0x0B77 },   .{ .lo = 0x0BF0, .hi = 0x0BF2 },   .{ .lo = 0x0C78, .hi = 0x0C7E },   .{ .lo = 0x0D58, .hi = 0x0D5E },
    .{ .lo = 0x0D70, .hi = 0x0D78 },   .{ .lo = 0x0F2A, .hi = 0x0F33 },   .{ .lo = 0x1369, .hi = 0x137C },   .{ .lo = 0x17F0, .hi = 0x17F9 },
    .{ .lo = 0x19DA, .hi = 0x19DA },   .{ .lo = 0x2070, .hi = 0x2070 },   .{ .lo = 0x2074, .hi = 0x2079 },   .{ .lo = 0x2080, .hi = 0x2089 },
    .{ .lo = 0x2150, .hi = 0x215F },   .{ .lo = 0x2189, .hi = 0x2189 },   .{ .lo = 0x2460, .hi = 0x249B },   .{ .lo = 0x24EA, .hi = 0x24FF },
    .{ .lo = 0x2776, .hi = 0x2793 },   .{ .lo = 0x2CFD, .hi = 0x2CFD },   .{ .lo = 0x3192, .hi = 0x3195 },   .{ .lo = 0x3220, .hi = 0x3229 },
    .{ .lo = 0x3248, .hi = 0x324F },   .{ .lo = 0x3251, .hi = 0x325F },   .{ .lo = 0x3280, .hi = 0x3289 },   .{ .lo = 0x32B1, .hi = 0x32BF },
    .{ .lo = 0xA830, .hi = 0xA835 },   .{ .lo = 0x10107, .hi = 0x10133 }, .{ .lo = 0x10175, .hi = 0x10178 }, .{ .lo = 0x1018A, .hi = 0x1018B },
    .{ .lo = 0x102E1, .hi = 0x102FB }, .{ .lo = 0x10320, .hi = 0x10323 }, .{ .lo = 0x10858, .hi = 0x1085F }, .{ .lo = 0x10879, .hi = 0x1087F },
    .{ .lo = 0x108A7, .hi = 0x108AF }, .{ .lo = 0x108FB, .hi = 0x108FF }, .{ .lo = 0x10916, .hi = 0x1091B }, .{ .lo = 0x109BC, .hi = 0x109BD },
    .{ .lo = 0x109C0, .hi = 0x109CF }, .{ .lo = 0x109D2, .hi = 0x109FF }, .{ .lo = 0x10A40, .hi = 0x10A48 }, .{ .lo = 0x10A7D, .hi = 0x10A7E },
    .{ .lo = 0x10A9D, .hi = 0x10A9F }, .{ .lo = 0x10AEB, .hi = 0x10AEF }, .{ .lo = 0x10B58, .hi = 0x10B5F }, .{ .lo = 0x10B78, .hi = 0x10B7F },
    .{ .lo = 0x10BA9, .hi = 0x10BAF }, .{ .lo = 0x10CFA, .hi = 0x10CFF }, .{ .lo = 0x10E60, .hi = 0x10E7E }, .{ .lo = 0x10F1D, .hi = 0x10F26 },
    .{ .lo = 0x10F51, .hi = 0x10F54 }, .{ .lo = 0x10FC5, .hi = 0x10FCB }, .{ .lo = 0x11052, .hi = 0x11065 }, .{ .lo = 0x111E1, .hi = 0x111F4 },
    .{ .lo = 0x1173A, .hi = 0x1173B }, .{ .lo = 0x118EA, .hi = 0x118F2 }, .{ .lo = 0x11C5A, .hi = 0x11C6C }, .{ .lo = 0x11FC0, .hi = 0x11FD4 },
    .{ .lo = 0x16B5B, .hi = 0x16B61 }, .{ .lo = 0x16E80, .hi = 0x16E96 }, .{ .lo = 0x1D2C0, .hi = 0x1D2D3 }, .{ .lo = 0x1D2E0, .hi = 0x1D2F3 },
    .{ .lo = 0x1D360, .hi = 0x1D378 }, .{ .lo = 0x1E8C7, .hi = 0x1E8CF }, .{ .lo = 0x1EC71, .hi = 0x1ECAB }, .{ .lo = 0x1ECAD, .hi = 0x1ECAF },
    .{ .lo = 0x1ECB1, .hi = 0x1ECB4 }, .{ .lo = 0x1ED01, .hi = 0x1ED2D }, .{ .lo = 0x1ED2F, .hi = 0x1ED3D }, .{ .lo = 0x1F100, .hi = 0x1F10C },
};

fn contains(ranges: []const Range, cp: u21) bool {
    var lo: usize = 0;
    var hi: usize = ranges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const r = ranges[mid];
        if (cp < r.lo) {
            hi = mid;
        } else if (cp > r.hi) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return false;
}

pub fn isLu(cp: u21) bool {
    return contains(&cat_Lu, cp);
}
pub fn isLl(cp: u21) bool {
    return contains(&cat_Ll, cp);
}
pub fn isLt(cp: u21) bool {
    return contains(&cat_Lt, cp);
}
pub fn isLm(cp: u21) bool {
    return contains(&cat_Lm, cp);
}
pub fn isLo(cp: u21) bool {
    return contains(&cat_Lo, cp);
}
pub fn isMn(cp: u21) bool {
    return contains(&cat_Mn, cp);
}
pub fn isMc(cp: u21) bool {
    return contains(&cat_Mc, cp);
}
pub fn isMe(cp: u21) bool {
    return contains(&cat_Me, cp);
}
pub fn isNd(cp: u21) bool {
    return contains(&cat_Nd, cp);
}
pub fn isNl(cp: u21) bool {
    return contains(&cat_Nl, cp);
}
pub fn isNo(cp: u21) bool {
    return contains(&cat_No, cp);
}
pub fn isCf(cp: u21) bool {
    return contains(&cat_Cf, cp);
}
pub fn isPunct(cp: u21) bool {
    return contains(&cat_P, cp);
}

// L = Lu | Ll | Lt | Lm | Lo
pub fn isLetter(cp: u21) bool {
    return isLl(cp) or isLu(cp) or isLo(cp) or isLm(cp) or isLt(cp);
}

// N = Nd | Nl | No
pub fn isNumber(cp: u21) bool {
    return isNd(cp) or isNl(cp) or isNo(cp);
}

// M = Mn | Mc | Me
pub fn isMark(cp: u21) bool {
    return isMn(cp) or isMc(cp) or isMe(cp);
}

// White_Space per PropList.txt (regex \s convention).
pub fn isWhitespace(cp: u21) bool {
    return (cp >= 0x09 and cp <= 0x0D) or
        cp == 0x20 or
        cp == 0x85 or
        cp == 0xA0 or
        cp == 0x1680 or
        (cp >= 0x2000 and cp <= 0x200A) or
        cp == 0x2028 or cp == 0x2029 or
        cp == 0x202F or
        cp == 0x205F or
        cp == 0x3000;
}

// ----- Combined character classifier ----------------------------------------
//
// Hot paths in capcode (`encodeStyled` / `NoCapcode.encode`) ask 3-5 of
// the `is*Cp` questions for the same codepoint on every iteration. Each
// query is a binary search over ~hundreds of ranges, so 4-5 queries cost
// ~40-50 compares per cp.
//
// `classifyCp` walks a SINGLE comptime-merged range table that returns a
// packed bitfield with all four predicates set in one lookup, replacing
// the per-class searches. ASCII has a precomputed `[128]CharClass`
// table — no binary search at all for the most common bytes.
//
// All four predicates (letter, number, mark, whitespace) are mutually
// exclusive on a given codepoint in Unicode 16.0 General_Category and
// White_Space (no cp is both letter and digit, no letter is whitespace,
// etc.) — but we still encode them as independent bits so future
// classes (punctuation, symbol, …) can be added without invalidating
// the existing bits.

pub const CharClass = packed struct(u8) {
    letter: bool = false,
    number: bool = false,
    mark: bool = false,
    whitespace: bool = false,
    _reserved: u4 = 0,

    pub fn empty() CharClass {
        return .{};
    }

    pub fn raw(self: CharClass) u8 {
        return @bitCast(self);
    }
};

const ClassRange = packed struct {
    lo: u24,
    hi: u24,
    klass: u8,
};

// Comptime bottom-up merge sort over a u32 slice. Used by the
// boundary-merge in `buildMergedTable`. Allocates no comptime memory
// itself — caller provides a same-size scratch buffer.
fn mergeSortInPlace(data: []u32, scratch: []u32) void {
    std.debug.assert(scratch.len >= data.len);
    var width: usize = 1;
    var src: []u32 = data;
    var dst: []u32 = scratch[0..data.len];
    while (width < data.len) : (width *= 2) {
        var i: usize = 0;
        while (i < data.len) : (i += 2 * width) {
            const mid: usize = @min(i + width, data.len);
            const end: usize = @min(i + 2 * width, data.len);
            var l: usize = i;
            var r: usize = mid;
            var o: usize = i;
            while (l < mid and r < end) {
                if (src[l] <= src[r]) {
                    dst[o] = src[l];
                    l += 1;
                } else {
                    dst[o] = src[r];
                    r += 1;
                }
                o += 1;
            }
            while (l < mid) : (l += 1) {
                dst[o] = src[l];
                o += 1;
            }
            while (r < end) : (r += 1) {
                dst[o] = src[r];
                o += 1;
            }
        }
        const tmp = src;
        src = dst;
        dst = tmp;
    }
    if (src.ptr != data.ptr) {
        for (data, 0..) |*p, idx| p.* = src[idx];
    }
}

// Walk every Range[] table and produce a sorted list of {lo, hi, klass}
// triples covering only the codepoints that belong to at least one class.
// Overlapping or adjacent same-class spans are coalesced.
fn buildMergedTable() []const ClassRange {
    @setEvalBranchQuota(200_000_000);

    // Per-class membership lists in `cat_*` order; mirrors isLetter /
    // isNumber / isMark / isWhitespace.
    const letter_tables = [_][]const Range{
        &cat_Lu, &cat_Ll, &cat_Lt, &cat_Lm, &cat_Lo,
    };
    const number_tables = [_][]const Range{ &cat_Nd, &cat_Nl, &cat_No };
    const mark_tables = [_][]const Range{ &cat_Mn, &cat_Mc, &cat_Me };
    // White_Space is a closed enum of disjoint singletons / tiny ranges
    // — encode it as a static table of Range so the same merge loop sees
    // it.
    const ws_table = [_]Range{
        .{ .lo = 0x0009, .hi = 0x000D },
        .{ .lo = 0x0020, .hi = 0x0020 },
        .{ .lo = 0x0085, .hi = 0x0085 },
        .{ .lo = 0x00A0, .hi = 0x00A0 },
        .{ .lo = 0x1680, .hi = 0x1680 },
        .{ .lo = 0x2000, .hi = 0x200A },
        .{ .lo = 0x2028, .hi = 0x2029 },
        .{ .lo = 0x202F, .hi = 0x202F },
        .{ .lo = 0x205F, .hi = 0x205F },
        .{ .lo = 0x3000, .hi = 0x3000 },
    };

    // Phase 1: count endpoints so we know how big the breakpoint list
    // needs to be at comptime.
    var total_ranges: usize = 0;
    for (letter_tables) |t| total_ranges += t.len;
    for (number_tables) |t| total_ranges += t.len;
    for (mark_tables) |t| total_ranges += t.len;
    total_ranges += ws_table.len;

    // Collect every endpoint into a sweep-line key set:
    //   for each [lo, hi], add lo and hi+1 as boundaries.
    // We then walk the sorted unique boundaries and for each
    // [boundary[i], boundary[i+1]-1] interval ask each predicate
    // whether the midpoint belongs.
    var boundaries: [total_ranges * 2 + 1]u32 = undefined;
    var nb: usize = 0;
    inline for (.{ letter_tables, number_tables, mark_tables }) |tables| {
        for (tables) |t| {
            for (t) |r| {
                boundaries[nb] = @intCast(r.lo);
                nb += 1;
                boundaries[nb] = @as(u32, @intCast(r.hi)) + 1;
                nb += 1;
            }
        }
    }
    for (ws_table) |r| {
        boundaries[nb] = @intCast(r.lo);
        nb += 1;
        boundaries[nb] = @as(u32, @intCast(r.hi)) + 1;
        nb += 1;
    }

    // Merge-sort the boundary array at comptime. O(n log n) compares,
    // ~70k for n=5300 vs ~14M for insertion-sort — keeps comptime
    // happy under a reasonable @setEvalBranchQuota.
    var bs = boundaries[0..nb];
    var scratch: [total_ranges * 2 + 1]u32 = undefined;
    mergeSortInPlace(bs, scratch[0..nb]);
    // Dedup in place.
    var w: usize = 0;
    var r: usize = 0;
    while (r < bs.len) {
        const v = bs[r];
        bs[w] = v;
        w += 1;
        while (r < bs.len and bs[r] == v) r += 1;
    }
    const nub = w;

    // Phase 2: build merged classified intervals. For each gap between
    // consecutive boundaries, classify its representative codepoint via
    // the existing binary-search tables, then coalesce adjacent gaps
    // that share the same class bitfield.
    var merged: [nub]ClassRange = undefined;
    var mn: usize = 0;
    var k: usize = 0;
    while (k + 1 < nub) : (k += 1) {
        const lo: u21 = @intCast(bs[k]);
        const hi: u21 = @intCast(bs[k + 1] - 1);
        var cls: u8 = 0;
        // isLetter
        var is_l = false;
        for (letter_tables) |t| {
            if (contains(t, lo)) {
                is_l = true;
                break;
            }
        }
        if (is_l) cls |= 0b0001;
        // isNumber
        var is_n = false;
        for (number_tables) |t| {
            if (contains(t, lo)) {
                is_n = true;
                break;
            }
        }
        if (is_n) cls |= 0b0010;
        // isMark
        var is_m = false;
        for (mark_tables) |t| {
            if (contains(t, lo)) {
                is_m = true;
                break;
            }
        }
        if (is_m) cls |= 0b0100;
        // isWhitespace (use the inline predicate so any future change to
        // isWhitespace stays the source of truth)
        if (isWhitespace(lo)) cls |= 0b1000;

        if (cls == 0) continue; // gap with no class — skip entirely.

        // Coalesce with the previous entry if class matches and they're
        // contiguous.
        if (mn > 0 and merged[mn - 1].klass == cls and merged[mn - 1].hi + 1 == lo) {
            merged[mn - 1].hi = hi;
        } else {
            merged[mn] = .{ .lo = lo, .hi = hi, .klass = cls };
            mn += 1;
        }
    }

    // Freeze the merged array into a compact const slice.
    var out: [mn]ClassRange = undefined;
    for (0..mn) |idx| out[idx] = merged[idx];
    const frozen = out;
    return &frozen;
}

pub const merged_class_table: []const ClassRange = buildMergedTable();

// Precomputed ASCII table for the common path. 128 entries.
pub const ascii_class_table: [128]CharClass = blk: {
    @setEvalBranchQuota(10_000);
    var table: [128]CharClass = undefined;
    for (0..128) |i| {
        const cp: u21 = @intCast(i);
        var c: CharClass = .{};
        if ((cp >= 'A' and cp <= 'Z') or (cp >= 'a' and cp <= 'z')) c.letter = true;
        if (cp >= '0' and cp <= '9') c.number = true;
        if ((cp >= 0x09 and cp <= 0x0D) or cp == 0x20 or cp == 0x85) c.whitespace = true;
        // No ASCII codepoint is a combining mark.
        table[i] = c;
    }
    break :blk table;
};

/// Classify `cp` into the four-bit `CharClass` bitfield — letter, number,
/// mark, whitespace. ASCII uses a flat 128-entry lookup; everything else
/// does a single binary search of the merged Unicode range table.
pub fn classifyCp(cp: u21) CharClass {
    if (cp < 0x80) return ascii_class_table[@intCast(cp)];
    var lo: usize = 0;
    var hi: usize = merged_class_table.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const r = merged_class_table[mid];
        if (cp < r.lo) {
            hi = mid;
        } else if (cp > r.hi) {
            lo = mid + 1;
        } else {
            return @bitCast(r.klass);
        }
    }
    return .{};
}

test "ascii letters and digits" {
    try std.testing.expect(isLetter('a'));
    try std.testing.expect(isLetter('Z'));
    try std.testing.expect(!isLetter('1'));
    try std.testing.expect(isNumber('5'));
    try std.testing.expect(!isNumber('a'));
}

test "greek case" {
    try std.testing.expect(isLl(0x03B1));
    try std.testing.expect(isLu(0x0391));
    try std.testing.expect(isLetter(0x03B1));
    try std.testing.expect(isLetter(0x0391));
}

test "cjk ideograph" {
    try std.testing.expect(isLo(0x4E00));
    try std.testing.expect(isLetter(0x4E00));
    try std.testing.expect(!isLu(0x4E00));
}

test "arabic-indic digits" {
    try std.testing.expect(isNd(0x0660));
    try std.testing.expect(isNumber(0x0660));
}

test "roman numerals are Nl not Nd" {
    try std.testing.expect(isNl(0x2160));
    try std.testing.expect(isNumber(0x2160));
    try std.testing.expect(!isNd(0x2160));
}

test "combining acute is a mark not a letter" {
    try std.testing.expect(isMn(0x0301));
    try std.testing.expect(isMark(0x0301));
    try std.testing.expect(!isLetter(0x0301));
}

test "whitespace" {
    try std.testing.expect(isWhitespace(' '));
    try std.testing.expect(isWhitespace(0x00A0));
    try std.testing.expect(isWhitespace(0x2028));
    try std.testing.expect(!isWhitespace('x'));
}

test "classifyCp ASCII fast path matches per-class predicates" {
    var cp: u21 = 0;
    while (cp < 128) : (cp += 1) {
        const c = classifyCp(cp);
        try std.testing.expectEqual(isLetter(cp), c.letter);
        try std.testing.expectEqual(isNumber(cp), c.number);
        try std.testing.expectEqual(isMark(cp), c.mark);
        try std.testing.expectEqual(isWhitespace(cp), c.whitespace);
    }
}

test "classifyCp equivalence over a wide deterministic codepoint sweep" {
    // Spot-check ~1000 codepoints spread across the BMP + a slice of
    // supplementary planes. Using a deterministic Wyhash so the test is
    // reproducible across runs.
    var rng: std.Random.DefaultPrng = .init(0x5345_4554);
    const r = rng.random();
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const cp: u21 = @intCast(r.intRangeAtMost(u32, 0, 0x2_FFFF));
        const c = classifyCp(cp);
        try std.testing.expectEqual(isLetter(cp), c.letter);
        try std.testing.expectEqual(isNumber(cp), c.number);
        try std.testing.expectEqual(isMark(cp), c.mark);
        try std.testing.expectEqual(isWhitespace(cp), c.whitespace);
    }
    // Hit a few hand-picked tricky cps for good measure.
    const hand = [_]u21{
        0x0041,  0x0061,  0x0030, 0x0020, 0x00A0, 0x00DC, 0x03B1,  0x03A3,
        0x041F,  0x0660,  0x2160, 0x0301, 0x4E00, 0x3041, 0x1F600, 0x10000,
        0x10428, 0x1F4A9, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,  0x1680,
    };
    for (hand) |cp| {
        const c = classifyCp(cp);
        try std.testing.expectEqual(isLetter(cp), c.letter);
        try std.testing.expectEqual(isNumber(cp), c.number);
        try std.testing.expectEqual(isMark(cp), c.mark);
        try std.testing.expectEqual(isWhitespace(cp), c.whitespace);
    }
}

test "merged table is smaller than the sum of the per-class tables" {
    // Regression check: the merge should NEVER blow up the total entry
    // count vs the sum of all per-class range tables (otherwise we'd
    // be paying more binary-search depth for the privilege).
    const per_class_total = cat_Lu.len + cat_Ll.len + cat_Lt.len +
        cat_Lm.len + cat_Lo.len + cat_Mn.len + cat_Mc.len + cat_Me.len +
        cat_Nd.len + cat_Nl.len + cat_No.len + 10; // 10 WS entries
    try std.testing.expect(merged_class_table.len < per_class_total);
}

test "classifyCp packed bits round-trip via @bitCast" {
    // Sanity: the packed struct must occupy exactly one byte.
    try std.testing.expectEqual(@as(usize, 1), @sizeOf(CharClass));
    var c: CharClass = .{};
    c.letter = true;
    c.number = true;
    const raw: u8 = @bitCast(c);
    try std.testing.expectEqual(@as(u8, 0b0011), raw);
}
