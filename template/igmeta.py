from igm.conf import igm_project, cpy, cpip

from config import DATA_SOURCE_FILE


def info():
    print('This is the project of clustering.')
    print(f'The native data is from file {DATA_SOURCE_FILE!r}')


igm_project(
    name={{user.title | potc}},
    version='0.0.1',
    template_name={{template.name | potc}},
    template_version={{template.version | potc}},
    created_at={{py.time.time() | potc}},
    scripts={
        None: cpy('main.py'),
        'info': info,
        'install': cpip('install', '-r', 'requirements.txt'),
    }
)
