# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import glob
import hashlib
import importlib
import importlib.util
import os
import shutil
import uuid
import traceback
import torch
import torch.utils.cpp_extension
import subprocess

#----------------------------------------------------------------------------
# Global options.

verbosity = 'brief' # Verbosity level: 'none', 'brief', 'full'

#----------------------------------------------------------------------------
# Internal helper funcs.

_msvc_env_activated = False

def _activate_msvc_env():
    global _msvc_env_activated
    if os.name != 'nt':
        return True
    if _msvc_env_activated or 'VSCMD_ARG_TGT_ARCH' in os.environ:
        return True
    if shutil.which('cl'):
        return True

    vswhere = r'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
    if not os.path.isfile(vswhere):
        return False

    try:
        result = subprocess.run(
            [vswhere, '-latest', '-products', '*',
             '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
             '-property', 'installationPath'],
            capture_output=True, text=True, check=True
        )
        vs_path = result.stdout.strip()
        if not vs_path:
            return False
    except Exception:
        return False

    vcvarsall = os.path.join(vs_path, 'VC', 'Auxiliary', 'Build', 'vcvarsall.bat')
    if not os.path.isfile(vcvarsall):
        return False

    try:
        out = subprocess.check_output(
            f'cmd /u /c "{vcvarsall}" x64 && set',
            stderr=subprocess.STDOUT,
        ).decode('utf-16le', errors='replace')
    except subprocess.CalledProcessError:
        return False

    for line in out.splitlines():
        key, _, value = line.partition('=')
        if key and value:
            os.environ[key] = value

    _msvc_env_activated = True
    return True



#----------------------------------------------------------------------------

def _get_cuda_compute_version():
    if not torch.cuda.is_available():
        return None
    return ".".join(map(str, torch.cuda.get_device_capability(torch.cuda.current_device())))

def _get_cache_identifier():
    arch_list_env = os.environ.get('TORCH_CUDA_ARCH_LIST', '').strip()

    if arch_list_env:
        arch_versions = [arch.strip() for arch in arch_list_env.split(';') if arch.strip()]
        if arch_versions:
            sorted_archs = sorted(arch_versions, key=lambda x: tuple(map(int, x.split('.'))))
            return '_'.join(sorted_archs)

    return _get_cuda_compute_version()

def _find_compatible_cache_dir(build_top_dir, source_digest):
    current_compute = _get_cuda_compute_version()

    if current_compute is None:
        return None

    cache_pattern = os.path.join(build_top_dir, f'{source_digest}-*')
    existing_caches = glob.glob(cache_pattern)

    if not existing_caches:
        return None

    for cache_dir in existing_caches:
        cache_suffix = os.path.basename(cache_dir).split('-', 1)[1]
        arch_list = cache_suffix.split('_')
        if current_compute in arch_list:
            return cache_dir

    return None

#----------------------------------------------------------------------------
# Main entry point for compiling and loading C++/CUDA plugins.

_cached_plugins = dict()

def get_plugin(module_name, sources, headers=None, source_dir=None, force_rebuild=False, **build_kwargs):
    assert verbosity in ['none', 'brief', 'full']
    if headers is None:
        headers = []
    if source_dir is not None:
        sources = [os.path.join(source_dir, fname) for fname in sources]
        headers = [os.path.join(source_dir, fname) for fname in headers]

    if not force_rebuild and module_name in _cached_plugins:
        return _cached_plugins[module_name]

    # Print status.
    if verbosity == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)
    verbose_build = (verbosity == 'full')

    # Compile and load.
    try: # pylint: disable=too-many-nested-blocks
        # Make sure we can find the necessary compiler binaries.
        if os.name == 'nt' and not _activate_msvc_env():
            print('Could not find MSVC installation. Install Visual Studio Build Tools with the C++ workload.')
            return None

        all_source_files = sorted(sources + headers)
        all_source_dirs = set(os.path.dirname(fname) for fname in all_source_files)
        if len(all_source_dirs) == 1: # and ('TORCH_EXTENSIONS_DIR' in os.environ):

            # Compute combined hash digest for all source files.
            hash_md5 = hashlib.md5()
            for src in all_source_files:
                with open(src, 'rb') as f:
                    hash_md5.update(f.read())

            # Select cached build directory name.
            source_digest = hash_md5.hexdigest()
            build_top_dir = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose_build) # pylint: disable=protected-access
            cache_identifier = _get_cache_identifier()
            cached_build_dir = os.path.join(build_top_dir, f'{source_digest}-{cache_identifier}')

            if not force_rebuild:
                compatible = _find_compatible_cache_dir(build_top_dir, source_digest)
                if compatible is not None:
                    cached_build_dir = compatible

            if os.path.isdir(cached_build_dir):
                compiled_file = None
                for ext in ['.pyd', '.so']:
                    candidate = os.path.join(cached_build_dir, f'{module_name}{ext}')
                    if os.path.isfile(candidate):
                        compiled_file = candidate
                        break

                if compiled_file is not None:
                    try:
                        spec = importlib.util.spec_from_file_location(module_name, compiled_file)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        if verbosity != 'none':
                            print(f'Loaded cached PyTorch plugin "{module_name}".')
                        _cached_plugins[module_name] = module
                        return module
                    except Exception as e:
                        if verbosity != 'none':
                            print(f'Cached plugin "{module_name}" failed to load ({e}), rebuilding...')
                else:
                    if verbosity != 'none':
                        print(f'Incomplete cache for plugin "{module_name}", deleting cache and rebuilding...')
                    shutil.rmtree(cached_build_dir)

            if not os.path.isdir(cached_build_dir):
                tmpdir = f'{build_top_dir}/srctmp-{uuid.uuid4().hex}'
                os.makedirs(tmpdir)
                for src in all_source_files:
                    shutil.copyfile(src, os.path.join(tmpdir, os.path.basename(src)))
                try:
                    os.replace(tmpdir, cached_build_dir) # atomic
                except OSError:
                    shutil.rmtree(tmpdir)
                    if not os.path.isdir(cached_build_dir): return False

            cached_sources = [os.path.join(cached_build_dir, os.path.basename(fname)) for fname in sources]
            torch.utils.cpp_extension.load(name=module_name, build_directory=cached_build_dir,
                verbose=verbose_build, sources=cached_sources, **build_kwargs)
        else:
            torch.utils.cpp_extension.load(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs)

        # Load.
        module = importlib.import_module(module_name)

    except Exception as e:
        if verbosity == 'brief':
            print('Failed!')
        print('Failed!')
        print(f"Error details: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

    # Print status and add to cache dict.
    if verbosity == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == 'brief':
        print('Done.')
    _cached_plugins[module_name] = module
    return module

#----------------------------------------------------------------------------
