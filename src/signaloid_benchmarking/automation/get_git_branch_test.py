#   Copyright (c) 2026, Signaloid.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to
#   deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.

"""
Coverage for the ``get_git_branch`` helper, exercised against real
throwaway Git repositories.
"""

import subprocess
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from signaloid_benchmarking.automation.benchmarking_utils import (
    get_git_branch,
)


def _run_git(cwd: Path, *args: str) -> None:
    """
    Run a Git command in ``cwd`` and fail loudly if it does not succeed.

    Args:
        cwd: Directory to run the command in.
        *args: Arguments passed to the ``git`` executable.
    """
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(path: Path, branch: str = "main") -> None:
    """
    Create a repository at ``path`` with one empty commit on ``branch``.

    Commit signing and the committer identity are set per command, so the
    test does not depend on the developer's global Git configuration.

    Args:
        path: Directory to turn into a repository.
        branch: Name of the initial branch.
    """
    _run_git(path, "init", "--quiet", "--initial-branch", branch)
    _run_git(
        path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "--allow-empty",
        "-m",
        "initial",
    )


class TestGetGitBranch(unittest.TestCase):
    """``get_git_branch`` names the checked-out branch, or returns
    ``None`` when there is no branch to name."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def test_returns_initial_branch_name(self) -> None:
        """A repository sitting on its initial branch reports that name."""
        repo = self.tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)

        self.assertEqual(get_git_branch(str(repo)), "main")

    def test_returns_name_of_checked_out_branch(self) -> None:
        """The reported name follows a checkout, not the initial branch."""
        repo = self.tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        _run_git(repo, "checkout", "--quiet", "-b", "issue-101")

        self.assertEqual(get_git_branch(str(repo)), "issue-101")

    def test_returns_branch_name_before_the_first_commit(self) -> None:
        """An unborn branch still has a name, so it is reported."""
        repo = self.tmp_path / "repo"
        repo.mkdir()
        _run_git(repo, "init", "--quiet", "--initial-branch", "main")

        self.assertEqual(get_git_branch(str(repo)), "main")

    def test_returns_branch_name_inside_a_worktree(self) -> None:
        """A linked worktree has a .git file rather than a .git directory.

        The helper must still name its branch, since the ``.git`` presence
        check would otherwise reject a perfectly valid checkout.
        """
        repo = self.tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        worktree = self.tmp_path / "worktree"
        _run_git(
            repo,
            "worktree",
            "add",
            "--quiet",
            "-b",
            "side-branch",
            str(worktree),
        )

        self.assertTrue((worktree / ".git").is_file())
        self.assertEqual(get_git_branch(str(worktree)), "side-branch")

    def test_returns_none_for_detached_head(self) -> None:
        """A detached HEAD, as CI checkouts produce, has no branch name."""
        repo = self.tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        _run_git(repo, "checkout", "--quiet", "--detach")

        self.assertIsNone(get_git_branch(str(repo)))

    def test_returns_none_when_not_a_git_repository(self) -> None:
        """A plain directory is reported as having no branch."""
        plain = self.tmp_path / "plain"
        plain.mkdir()

        self.assertIsNone(get_git_branch(str(plain)))

    def test_returns_none_for_a_missing_directory(self) -> None:
        """A path that does not exist is reported as having no branch."""
        missing = self.tmp_path / "does-not-exist"

        self.assertIsNone(get_git_branch(str(missing)))

    def test_returns_none_when_git_is_unavailable(self) -> None:
        """A missing ``git`` executable is tolerated without raising."""
        repo = self.tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)

        with patch.object(
            subprocess,
            "check_output",
            side_effect=FileNotFoundError(2, "No such file or directory"),
        ):
            self.assertIsNone(get_git_branch(str(repo)))


if __name__ == "__main__":
    unittest.main()
