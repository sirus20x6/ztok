/*
 * One token window produced by {@link Pipeline#chunk}. Mirrors
 * ztok_chunk_rec from include/ztok.h, with the id buffer copied out of
 * native memory into a Java int[] (the native buffer is freed before the
 * Chunk is handed back, so this object is fully JVM-owned).
 *
 * {@code ids} are the token ids in this chunk. {@code byteStart}/
 * {@code byteEnd} is the half-open byte range the chunk covers in the
 * ORIGINAL input; {@code tokenStart}/{@code tokenEnd} the half-open
 * token-index range in the full encoding.
 */
package com.anthropic.ztok;

public record Chunk(
    int[] ids,
    int byteStart,
    int byteEnd,
    int tokenStart,
    int tokenEnd
) {}
