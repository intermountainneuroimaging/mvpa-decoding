"""hcp_resample.py: warp-file path resolution and applywarp command
construction. Synthetic/mocked only -- no dependency on FSL being installed
(CI has no FSL, same reasoning as tutorial/preprocess_haxby.sh not being
part of the automated suite)."""

import pytest

from hcp_resample import (
    resolve_xfm_path,
    build_applywarp_command,
    check_fsl_available,
    run_applywarp,
)


# =====================================================
# resolve_xfm_path
# =====================================================

class TestResolveXfmPath:
    def _make_xfm(self, tmp_path, subject, session, filename):
        if session is not None:
            xfm_dir = tmp_path / f"sub-{subject}" / f"ses-{session}" / "MNINonLinear" / "xfms"
        else:
            xfm_dir = tmp_path / f"sub-{subject}" / "MNINonLinear" / "xfms"
        xfm_dir.mkdir(parents=True)
        path = xfm_dir / filename
        path.write_text("fake warp")
        return path

    def test_mni2native_with_session(self, tmp_path):
        expected = self._make_xfm(tmp_path, "001", "D1", "standard2acpc_dc.nii.gz")
        result = resolve_xfm_path(str(tmp_path), "001", "D1", "mni2native")
        assert result == str(expected)

    def test_native2mni_with_session(self, tmp_path):
        expected = self._make_xfm(tmp_path, "001", "D1", "acpc_dc2standard.nii.gz")
        result = resolve_xfm_path(str(tmp_path), "001", "D1", "native2mni")
        assert result == str(expected)

    def test_mni2native_without_session(self, tmp_path):
        expected = self._make_xfm(tmp_path, "1", None, "standard2acpc_dc.nii.gz")
        result = resolve_xfm_path(str(tmp_path), "1", None, "mni2native")
        assert result == str(expected)

    def test_custom_xfm_pattern_override(self, tmp_path):
        custom_dir = tmp_path / "custom" / "001"
        custom_dir.mkdir(parents=True)
        path = custom_dir / "my_warp.nii.gz"
        path.write_text("fake warp")

        result = resolve_xfm_path(
            str(tmp_path), "001", None, "mni2native",
            xfm_pattern="custom/{subject}/my_warp.nii.gz",
        )
        assert result == str(path)

    def test_missing_file_raises_with_resolved_path_in_message(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            resolve_xfm_path(str(tmp_path), "999", "D1", "mni2native")
        assert "sub-999" in str(exc_info.value)
        assert "standard2acpc_dc.nii.gz" in str(exc_info.value)

    def test_invalid_direction_raises(self, tmp_path):
        with pytest.raises(SystemExit):
            resolve_xfm_path(str(tmp_path), "001", "D1", "sideways")


# =====================================================
# build_applywarp_command
# =====================================================

class TestBuildApplywarpCommand:
    def test_minimal_command(self):
        cmd = build_applywarp_command("in.nii.gz", "out.nii.gz", "ref.nii.gz", "warp.nii.gz")
        assert cmd == [
            "applywarp",
            "--in=in.nii.gz",
            "--ref=ref.nii.gz",
            "--warp=warp.nii.gz",
            "--out=out.nii.gz",
            "--interp=trilinear",
        ]

    def test_custom_interp(self):
        cmd = build_applywarp_command("in.nii.gz", "out.nii.gz", "ref.nii.gz", "warp.nii.gz", interp="nn")
        assert "--interp=nn" in cmd

    def test_datatype_included_when_given(self):
        cmd = build_applywarp_command("in.nii.gz", "out.nii.gz", "ref.nii.gz", "warp.nii.gz", datatype="int")
        assert "--datatype=int" in cmd

    def test_datatype_omitted_when_none(self):
        cmd = build_applywarp_command("in.nii.gz", "out.nii.gz", "ref.nii.gz", "warp.nii.gz")
        assert not any(c.startswith("--datatype") for c in cmd)

    def test_warp_convention_included_when_given(self):
        cmd = build_applywarp_command("in.nii.gz", "out.nii.gz", "ref.nii.gz", "warp.nii.gz", warp_convention="rel")
        assert "--rel" in cmd

    def test_warp_convention_omitted_by_default(self):
        cmd = build_applywarp_command("in.nii.gz", "out.nii.gz", "ref.nii.gz", "warp.nii.gz")
        assert "--rel" not in cmd
        assert "--abs" not in cmd


# =====================================================
# check_fsl_available
# =====================================================

class TestCheckFslAvailable:
    def test_raises_when_applywarp_not_on_path(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        with pytest.raises(SystemExit) as exc_info:
            check_fsl_available()
        assert "module load fsl" in str(exc_info.value)

    def test_passes_when_applywarp_on_path(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/local/fsl/bin/applywarp")
        check_fsl_available()  # no exception


# =====================================================
# run_applywarp
# =====================================================

class FakeCompletedProcess:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestRunApplywarp:
    def test_raises_with_stderr_on_nonzero_exit(self, monkeypatch):
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, capture_output, text: FakeCompletedProcess(1, stderr="something went wrong"),
        )
        with pytest.raises(SystemExit) as exc_info:
            run_applywarp(["applywarp", "--in=x"])
        assert "something went wrong" in str(exc_info.value)

    def test_no_exception_on_success(self, monkeypatch):
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, capture_output, text: FakeCompletedProcess(0),
        )
        run_applywarp(["applywarp", "--in=x"])  # no exception
