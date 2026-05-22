// Shared helper for fuzz-nightly.yml: file (or de-dup) a GitHub Issue
// when a fuzz harness fails. Idempotent — if an open issue with the
// same title already exists, post a comment with the new run link
// instead of opening a duplicate.
//
// Used by `actions/github-script@v7` steps via `require(...)` against
// the workspace path. Exports a single async function.

module.exports = async function fileFuzzIssue({
  github,
  context,
  title,
  seed,
  job,
  repro,
  logPath,
}) {
  const fs = require('fs');

  // Read the last 200 lines of the captured log (if present) so the
  // issue body has a self-contained failure excerpt without needing
  // the run artifacts. Cap at ~64 KB to stay well under GitHub's
  // 65,536-char issue body limit.
  let logTail = '(log file missing)';
  try {
    const raw = fs.readFileSync(logPath, 'utf8');
    const lines = raw.split('\n');
    const tail = lines.slice(-200).join('\n');
    logTail = tail.length > 60000 ? tail.slice(-60000) : tail;
  } catch (e) {
    logTail = `(failed to read log ${logPath}: ${e.message})`;
  }

  const runUrl = `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`;
  const body = [
    `**Job:** \`${job}\``,
    `**Seed:** \`${seed}\``,
    `**Run:** ${runUrl}`,
    `**Repro:**`,
    '',
    '```sh',
    repro,
    '```',
    '',
    `**Log tail (last 200 lines):**`,
    '',
    '```',
    logTail,
    '```',
  ].join('\n');

  // Idempotent: search OPEN issues with the fuzz-failure label whose
  // title matches exactly. If one exists, append a comment with the
  // new run link instead of opening a duplicate.
  const existing = await github.rest.issues.listForRepo({
    owner: context.repo.owner,
    repo: context.repo.repo,
    state: 'open',
    labels: 'fuzz-failure',
    per_page: 100,
  });
  const dup = existing.data.find((iss) => iss.title === title);
  if (dup) {
    await github.rest.issues.createComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: dup.number,
      body: `Re-failed on nightly run: ${runUrl}\n\n<details><summary>log tail</summary>\n\n\`\`\`\n${logTail}\n\`\`\`\n\n</details>`,
    });
    return;
  }

  await github.rest.issues.create({
    owner: context.repo.owner,
    repo: context.repo.repo,
    title,
    body,
    labels: ['fuzz-failure'],
  });
};
