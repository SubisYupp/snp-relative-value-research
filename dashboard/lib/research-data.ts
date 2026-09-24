/** Lossless deployment transport; local research JSON remains unchanged. */
export async function fetchResearch(path: string, init?: RequestInit): Promise<Response> {
  if (process.env.NEXT_PUBLIC_PACKED_DATA !== '1') return fetch(path, init);
  const response = await fetch(`${path}.bin`, init);
  if (!response.ok || !response.body) return response;
  const decoded = new Response(response.body.pipeThrough(new DecompressionStream('gzip')));
  const value = await decoded.json();
  if (value && value._format === 'columnar-snapshot-v1') {
    value.rows = value.values.map((row: unknown[]) =>
      Object.fromEntries(value.columns.map((key: string, index: number) => [key, row[index]]))
    );
    delete value.values;
    delete value.columns;
    delete value._format;
  }
  return new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } });
}
