import os
import platform
import re
import subprocess
import sys
import types
import yaml
from datetime import datetime
from pathlib import Path

def get_platform_info():
    info = dict()
    info['system'] = platform.system()
    info['version'] = platform.version()
    info['release'] = platform.release()
    info['processor'] = platform.processor()
    info['machine'] = platform.machine()
    info['architecture'] = platform.architecture()[0]
    info['CPU_cores'] = platform.os.cpu_count()

    return {'platform' : info}


def get_python_info():
    info = dict()
    major, minor, micro, *others = tuple(sys.version_info)
    try:
        releaselevel = others[0]
        serial = others[1]
    except IndexError :
        releaselevel = None
        serial = None

    version_full = re.sub(r'[\r\n]', '', sys.version)
    version_full = re.sub(r'\s+', ' ', version_full).strip()

    info['version'] = f"{major}.{minor}.{micro}"
    info['version_full'] = version_full
    info['build'] = ' '.join( platform.python_build() )
    info['compiler'] = platform.python_compiler()
    info['implementation'] = sys.implementation.name
    if releaselevel:
        info['releaselevel'] = releaselevel
    if serial is not None:
        info['serial'] = serial

    return {'python' : info }

def get_modules_info():
    env_vars = locals()
    modules = {k : v for k, v in env_vars.items() if isinstance(v, types.ModuleType) and ('builtin' not in k) }
    return { 'modules' : modules}


def get_conda_info():
    info = dict()
    info['env_name'] = os.environ.get('CONDA_DEFAULT_ENV', 'NA')
    info['path'] = os.environ.get('CONDA_PREFIX', 'NA')
    return { 'conda' : info }


def get_cuda_info():
    info = dict()
    try :
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"
        ], encoding='utf-8' )
        out = re.sub(r'\s+', ' ', out)
        name, mem_totoal, mem_used, utilization = out.split(',')
        info = {
            'name' : name.strip(),
            'memory_total_MB' : int(mem_totoal),
            'memory_used_MB' : int(mem_used),
            'utilization_percent' : int(utilization)
        }
    except FileNotFoundError:
        info['error'] = 'nvidia-smi não encontrado. Certifique-se que os drivers NVIDIA estão instalados!'
    except Exception :
        info['error'] = 'Erro ao obter informações da GPU'

    return {'cuda' : info }

def _run_shell(cmd):

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        text=True,
        encoding='utf-8'
    )

    stdout, stderr = process.communicate()

    # stdout sempre vai ser str pq text=True
    result = {
        'stdout' : stdout.strip() if isinstance(stdout, str) else stdout,
        'stderr' : stderr.strip() if isinstance(stderr, str) else stderr,
        'returncode' : process.returncode
    }
    return result

def _get_git_username():
    label = None
    result = _run_shell(['git', 'config', '--get', 'user.name'])
    if result['returncode'] == 0:
        label = result['stdout']
    return label

def _get_git_user_email():
    label = None
    result = _run_shell(['git', 'config', '--get', 'user.email'])
    if result['returncode'] == 0:
        label = result['stdout']
    return label

def _get_git_version():
    label = None
    result = _run_shell(['git', '--version'])
    if result['returncode'] == 0 :
        label = result['stdout']
    return label

def _get_git_remote_origin():
    label = None
    result = _run_shell(['git', 'config', '--get', 'remote.origin.url'])
    if result['returncode'] == 0:
        label = result['stdout']
    return label

def _get_git_commit_hash():
    label = None
    result = _run_shell(['git', 'rev-parse', 'HEAD'])
    if result['returncode'] == 0:
        label = result['stdout']
    return label

def _get_git_branch():
    label = None
    result = _run_shell(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    if result['returncode'] == 0:
        label = result['stdout']
    return label

def _get_git_current_status():
    label = None
    result = _run_shell(['git', 'status', '--porcelain'])
    if result['returncode'] == 0:
        label = result['stdout']
    return label

def get_git_repository_info():
    info = {
        'username' : _get_git_username(),
        'email' : _get_git_user_email(),
        'branch'   : _get_git_branch(),
        'url'  : _get_git_remote_origin(),
        'version'  : _get_git_version(),
        'head' : _get_git_commit_hash(),
        # 'status' : _get_git_current_status()
    }
    return {'git' : info }

def get_torch_info():
    info = dict()
    try :
        import torch
    except ImportError:
        info['erro'] = 'Não foi possível importar a biblioteca torch'
    else:
        info = {
            'version' : f"{ torch.__version__}",
            'cuda_is_available' : torch.cuda.is_available(),
            'cuda_bf16_supported' : torch.cuda.is_bf16_supported(),
            'cuda_version' : torch.version.cuda,
            'device_count' : torch.cuda.device_count(),
            'num_threads' : torch.get_num_threads(),
        }

        devices = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
        devices = [ {
                        'name' : p.name,
                        'total_memory' : p.total_memory,
                        'version' : f'{p.major}.{p.minor}',
                        'l2_cache_size' : p.L2_cache_size,
                        'multi_processor_count' : p.multi_processor_count
                    } for p in devices ]
        info['devices'] = devices

    return {'torch' : info}


def env_report(
            dest='.',
            filename='envsinfo.yaml',
            with_conda_info=True,
            with_cuda_info=True,
            with_git_info=True,
            with_platform_info=True,
            with_python_info=True,
            with_torch_info=False,
    ):

    info = dict()
    info['_timestamp'] = datetime.now().isoformat()

    if with_platform_info:
        info.update( get_platform_info() )
    if with_python_info:
        info.update( get_python_info() )
    if with_conda_info :
        info.update( get_conda_info() )
    if with_cuda_info :
        info.update( get_cuda_info() )
    if with_git_info :
        info.update( get_git_repository_info() )
    if with_torch_info :
        info.update( get_torch_info() )

    filepath = Path(dest, filename)

    with open(filepath, 'w') as file :
        txt = yaml.dump(info)
        file.write(txt)

    return filepath
