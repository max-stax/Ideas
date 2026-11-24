# Ideas

A home-screen app for dumping thoughts and seeing what connects.

- **Dump**: type and save. Search everything. `#words` work as tags if you want them.
- **Themes**: groups of thoughts that mean similar things, found on your phone. Plus "Claude's read" when Claude has written one.
- **Resurface**: an old thought and what sits near it, plus a "collision" of two loosely related thoughts.
- **Related**: tap any thought to see the closest ones.

## How it works

- The app is static, served by GitHub Pages from this public repo. It holds no thoughts.
- Thoughts are saved on your phone and sync to the private `idea-bank` repo as `thoughts.json`.
- Related thoughts and themes come from a small language model (all-MiniLM-L6-v2, about 25 MB) that runs in the browser via Transformers.js. It downloads once, then works offline. Your text never leaves the phone for this.

## Setup (one time)

1. **Turn on Pages:** Settings → Pages → Source: *Deploy from a branch* → `main` / `(root)` → Save.
2. **Token:** GitHub → Settings → Developer settings → Fine-grained tokens → Generate. Repository access: only `idea-bank`. Permissions: *Contents: Read and write*. Paste it in the app under Settings.
3. **Install:** open the Pages URL in Safari → Share → *Add to Home Screen*.
