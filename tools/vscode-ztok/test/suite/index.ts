// Mocha bootstrap for the in-process suite.

import * as path from 'path';
import Mocha from 'mocha';
import * as fs from 'fs';

export function run(): Promise<void> {
  const mocha = new Mocha({ ui: 'tdd', color: false, timeout: 60_000 });
  const testsRoot = __dirname;

  return new Promise((resolve, reject) => {
    try {
      const files = fs
        .readdirSync(testsRoot)
        .filter((f) => f.endsWith('.test.js'))
        .map((f) => path.resolve(testsRoot, f));
      for (const f of files) mocha.addFile(f);
      mocha.run((failures) => {
        if (failures > 0) reject(new Error(`${failures} test(s) failed`));
        else resolve();
      });
    } catch (e) {
      reject(e instanceof Error ? e : new Error(String(e)));
    }
  });
}
