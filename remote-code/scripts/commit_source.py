"""Supervisor-invoked, metadata-only source commit; never pushes or installs Git.

Run on the worker after freezing source, then preserve the entire .git directory
and code_provenance.json with that exact source snapshot. Existing repositories
are refused unless this script created and marked them. Subsequent source changes
require explicit --update; default invocation verifies the original commit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

SOURCE_DIRS = ('src', 'scripts', 'tests', 'configs')
SOURCE_FILES = ('pyproject.toml', 'requirements-worker.txt', 'dependency-lock-or-export.txt',
                'README.md', 'DECISIONS.md', 'validation.md', 'learner-review.md', 'market-review.md')
MARKER = 'codex-source-provenance.json'
OWNER = {'owner': 'Codex source provenance', 'version': 1}


def git(root, *args, check=True):
    environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    environment.update(GIT_AUTHOR_NAME='Codex', GIT_AUTHOR_EMAIL='codex@localhost',
                       GIT_COMMITTER_NAME='Codex', GIT_COMMITTER_EMAIL='codex@localhost')
    return subprocess.run(['git', '-C', str(root), *args], check=check, text=True, env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def paths(root):
    found = []
    for folder in SOURCE_DIRS:
        found.extend(path for path in (root / folder).rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc')
    found.extend(root / name for name in SOURCE_FILES if (root / name).is_file())
    if any(path.is_symlink() for path in found):
        raise RuntimeError('Refusing symlinked source files; freeze explicit regular source files')
    return sorted(set(found))


def snapshot(root):
    return {path.relative_to(root).as_posix(): digest(path) for path in paths(root)}


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def main(root, update=False):
    if shutil.which('git') is None:
        raise RuntimeError('Git is unavailable; supervisor must supply it on the worker. No install attempted.')
    root = root.resolve()
    if not (root / 'src').is_dir() or not (root / 'pyproject.toml').is_file():
        raise RuntimeError('Expected a project root containing src/ and pyproject.toml')
    git_dir = root / '.git'
    if git_dir.exists():
        if not git_dir.is_dir() or git_dir.is_symlink():
            raise RuntimeError('Refusing an existing worktree or symlinked Git directory')
        marker = git_dir / MARKER
        if not marker.exists() or json.loads(marker.read_text()) != OWNER:
            raise RuntimeError('Refusing to modify a repository not created by this script')
    else:
        ancestor = git(root, 'rev-parse', '--show-toplevel', check=False)
        if ancestor.returncode == 0:
            raise RuntimeError('Project is inside an existing repository; no nested repository created')
        git(root, 'init', '--quiet')
        atomic_json(git_dir / MARKER, OWNER)
    before = snapshot(root)
    tracked = git(root, 'ls-files', '-z').stdout.split('\0')
    foreign = set(filter(None, tracked)) - set(before)
    # A removed previously tracked source file is safe only with explicit update.
    disallowed = {name for name in foreign if name not in SOURCE_FILES and
                  name.split('/')[0] not in SOURCE_DIRS}
    if disallowed:
        raise RuntimeError('Repository contains unexpected tracked files: ' + ', '.join(sorted(disallowed)))
    head = git(root, 'rev-parse', '--verify', 'HEAD', check=False)
    previous = root / 'code_provenance.json'
    old_record = json.loads(previous.read_text()) if previous.exists() else None
    if head.returncode == 0 and not update:
        if old_record is None:
            raise RuntimeError('Existing managed commit lacks provenance; inspect it before explicit --update')
        if old_record.get('commit') != head.stdout.strip() or old_record.get('tracked_source_sha256') != canonical_hash(before):
            raise RuntimeError('Frozen source or HEAD changed; supervisor must inspect and explicitly use --update')
        if git(root, 'diff', 'HEAD', '--exit-code', check=False).returncode:
            raise RuntimeError('Tracked source differs from frozen HEAD')
        print(json.dumps(old_record, sort_keys=True))
        return
    # Add only the named source/documentation scope, including allowed removals.
    names = sorted(set(before) | foreign)
    git(root, 'add', '--', *names)
    if snapshot(root) != before:
        raise RuntimeError('Source changed during staging; freeze all source writers and retry')
    staged_names = set(filter(None, git(root, 'ls-files', '-z').stdout.split('\0')))
    if staged_names != set(before):
        raise RuntimeError('Git index does not exactly match the frozen source file manifest')
    changed = git(root, 'diff', '--cached', '--quiet', check=False).returncode != 0
    if head.returncode != 0 or changed:
        git(root, '-c', 'user.name=Codex', '-c', 'user.email=codex@localhost',
            '-c', 'core.hooksPath=/dev/null', '-c', 'commit.gpgsign=false',
            'commit', '--quiet', '-m', 'Freeze synthetic deep hedging campaign source')
    if snapshot(root) != before or git(root, 'diff', 'HEAD', '--exit-code', check=False).returncode:
        raise RuntimeError('Source changed during commit; provenance was not published')
    commit = git(root, 'rev-parse', 'HEAD').stdout.strip()
    # Match src.protocol.source_hash without importing its numerical dependency.
    runtime_manifest = {name: value for name, value in before.items()
                        if name.split('/')[0] in ('src', 'scripts', 'configs')}
    record = dict(commit=commit, git_tree=git(root, 'rev-parse', 'HEAD^{tree}').stdout.strip(),
                  tracked_source_sha256=canonical_hash(before), source_hash=canonical_hash(runtime_manifest),
                  files=before, identity='Codex <codex@localhost>',
                  scope='Local source-only repository; no remote configured or push performed')
    atomic_json(previous, record)
    print(json.dumps(record, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--update', action='store_true', help='Explicitly commit reviewed source changes in this script-owned repository')
    arguments = parser.parse_args()
    main(arguments.root, arguments.update)
