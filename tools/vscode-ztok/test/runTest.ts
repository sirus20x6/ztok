// Boots VS Code via @vscode/test-electron and runs the in-process suite.
//
// Usage: `npm test`.
//
// We download a clean VS Code (cached in `.vscode-test/`) and launch it
// pointing at this extension folder, then run `suite/index.js` inside.

import * as path from 'path';
import { runTests } from '@vscode/test-electron';

async function main(): Promise<void> {
  try {
    const extensionDevelopmentPath = path.resolve(__dirname, '..', '..');
    const extensionTestsPath = path.resolve(__dirname, './suite/index.js');
    await runTests({
      extensionDevelopmentPath,
      extensionTestsPath,
      launchArgs: ['--disable-extensions', '--disable-gpu'],
    });
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error('VS Code test run failed:', err);
    process.exit(1);
  }
}

void main();
