// Browser launch shared by the tools. Default: the installed Chrome (channel 'chrome').
// Environment overrides, for machines without Chrome or without a GPU (e.g. a cloud container):
//   CHROME=/path/to/chrome or headless_shell   (default there: a headless shell found under
//                                              $PLAYWRIGHT_BROWSERS_PATH, if any)
//   SWIFTSHADER=1  real WebGPU on the CPU (SwiftShader through ANGLE/Vulkan): correct images and
//                  validation errors, but seconds per frame; timings mean nothing
//   FAKEGPU=1      stub WebGPU in every page (tools/fakegpu.js): UI and main-thread logic only,
//                  nothing is rendered
//   CHROME_ARGS="--flag ..." extra command-line flags
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright-core';

const HERE = path.dirname(new URL(import.meta.url).pathname);
export const FAKEGPU_SCRIPT = path.join(HERE, 'fakegpu.js');

export function findChrome() {
  if (process.env.CHROME) return process.env.CHROME;
  const root = process.env.PLAYWRIGHT_BROWSERS_PATH;
  if (!root || !fs.existsSync(root)) return undefined;
  for (const d of fs.readdirSync(root).filter((x) => x.startsWith('chromium_headless_shell')).sort().reverse()) {
    for (const p of ['chrome-linux/headless_shell', 'chrome-headless-shell-linux64/chrome-headless-shell']) if (fs.existsSync(path.join(root, d, p))) return path.join(root, d, p);
  }
  return undefined;
}

const SWIFTSHADER_ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE', '--use-vulkan=swiftshader'];

// launch(options): like chromium.launch; `args` are added to. With FAKEGPU=1 every new page
// (browser.newPage / browser.newContext) gets the stub WebGPU.
export async function launch({ args = [], headless = true, ...rest } = {}) {
  const exe = process.env.CHROME || (process.env.SWIFTSHADER || process.env.FAKEGPU ? findChrome() : undefined);
  const extra = [...(process.env.SWIFTSHADER ? SWIFTSHADER_ARGS : []), ...(process.env.CHROME_ARGS || '').split(/\s+/).filter(Boolean)];
  const browser = await chromium.launch({ ...(exe ? { executablePath: exe } : { channel: 'chrome' }), headless, args: [...args, ...extra], ...rest });
  if (process.env.FAKEGPU) {
    const newContext = browser.newContext.bind(browser);
    browser.newContext = async (o) => { const c = await newContext(o); await c.addInitScript({ path: FAKEGPU_SCRIPT }); return c; };
    browser.newPage = async (o) => { const c = await browser.newContext(o); const p = await c.newPage(); p.on('close', () => c.close().catch(() => {})); return p; };
  }
  return browser;
}
