import { readdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";

const root = new URL("../dist/", import.meta.url);
async function walk(prefix = "") {
  const entries = await readdir(new URL(prefix, root), { withFileTypes: true });
  const result = [];
  for (const entry of entries) {
    const name = prefix + entry.name;
    if (entry.isDirectory()) result.push(...(await walk(name + "/")));
    else if (name !== "sw.js") result.push(name);
  }
  return result.sort();
}
const files = await walk();
const hash = createHash("sha256");
for (const file of files) hash.update(await readFile(new URL(file, root)));
const version = hash.digest("hex").slice(0, 16);
await writeFile(
  new URL("sw.js", root),
  `
const CACHE = 'voicebox-kws-shell-${version}';
const FILES = ${JSON.stringify(files)};
const urls = FILES.map(file => new URL(file, self.registration.scope).href);
self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(urls)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', event => {
  event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key.startsWith('voicebox-kws-shell-') && key !== CACHE).map(key => caches.delete(key)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin || !url.href.startsWith(self.registration.scope)) return;
  if (event.request.mode === 'navigate') {
    event.respondWith(fetch(event.request).catch(() => caches.open(CACHE).then(cache => cache.match(new URL('index.html', self.registration.scope), { ignoreVary: true }))));
  } else if (urls.includes(url.href)) {
    // The server may add Vary: Origin; anonymous module requests and precache
    // requests can have different Origin headers. These are public static files.
    event.respondWith(caches.open(CACHE).then(async cache => (await cache.match(event.request, { ignoreVary: true })) || fetch(event.request)));
  }
});
`,
);
console.log(`Offline app shell: ${files.length} assets, version ${version}`);
