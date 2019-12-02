#-----------------------------------------------------------------------------
# Copyright (c) 2012 - 2019, Anaconda, Inc., and Bokeh Contributors.
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
#
# Adapted from bokeh for ArviZ purposes
#-----------------------------------------------------------------------------
""" Generate a gallery of Bokeh plots from a configuration file.

"""

#-----------------------------------------------------------------------------
# Boilerplate
#-----------------------------------------------------------------------------
import logging # isort:skip
log = logging.getLogger(__name__)

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Standard library imports
import json
import os
from os.path import abspath, dirname, exists, getmtime, isdir, isfile, join

# External imports
from sphinx.errors import SphinxError
from sphinx.util import ensuredir, status_iterator
from jinja2 import Environment, BaseLoader

# Bokeh imports
from bokeh.sphinxext.bokeh_directive import BokehDirective
from bokeh.sphinxext.templates import GALLERY_DETAIL
from bokeh.io import export_png

#-----------------------------------------------------------------------------
# Globals and constants
#-----------------------------------------------------------------------------


__all__ = (
    'BokehGalleryDirective',
    'setup',
)

EXPORT_CODE = """\n
export_png(ax, "{filename}")
"""

GALLERY_PAGE_RST = """{% for name in names %}
* |{{ name }}|
{% endfor %}

{% for name in names %}
.. |{{ name }}| image:: ../bokeh/_images/{{ name }}.png
    :target: bokeh/{{ name }}.html
    :class: gallery
{% endfor %}
"""

GALLERY_PAGE = Environment(loader=BaseLoader()).from_string(GALLERY_PAGE_RST)

#-----------------------------------------------------------------------------
# General API
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Dev API
#-----------------------------------------------------------------------------

class BokehGalleryDirective(BokehDirective):

    has_content = False
    required_arguments = 1

    def run(self):
        source_dir = self.state.document.settings.env.srcdir

        gallery_file = join(source_dir, self.arguments[0])

        gallery_dir = join(source_dir, "examples", "bokeh")
        if not exists(gallery_dir) and isdir(gallery_dir):
            raise SphinxError("gallery dir %r missing for gallery file %r" % (gallery_dir, gallery_file))

        spec = json.load(open(gallery_file))
        names = [detail['name']for detail in spec['details']]

        rst_text = GALLERY_PAGE.render(names=names)

        return self._parse(rst_text, "<bokeh-gallery>")

def config_inited_handler(app, config):
    gallery_dir = join(app.srcdir, config.bokeh_gallery_dir)
    gallery_file = join(gallery_dir, "gallery.json")
    gallery_rst = join(app.srcdir, "examples", "bokeh")

    if not exists(gallery_file) and isfile(gallery_file):
        raise SphinxError("could not find gallery file %r for configured gallery dir %r" % (gallery_file, gallery_dir))

    gallery_file_mtime = getmtime(gallery_file)

    ensuredir(gallery_rst)

    # we will remove each file we process from this set and see if anything is
    # left at the end (and remove it in that case)
    extras = set(os.listdir(gallery_dir))

    # app.env.note_dependency(specpath)
    spec = json.load(open(gallery_file))
    details = spec['details']

    names = set(x['name'] for x in details)
    if len(names) < len(details):
        raise SphinxError("gallery file %r has duplicate names" % gallery_file)

    details_iter = status_iterator(details,
                                   'creating gallery file entries... ',
                                   'brown',
                                   len(details),
                                   app.verbosity,
                                   stringify_func=lambda x: x['name'] + ".rst")

    images_dir = abspath(join(app.outdir, "examples", "bokeh", "_images"))
    ensuredir(images_dir)

    for detail in details_iter:
        detail_file_name = detail['name'] + ".rst"
        detail_file_path = join(gallery_rst, detail_file_name)

        if detail_file_path in extras:
            extras.remove(detail_file_path)

        # if the gallery detail file is newer than the gallery file, assume it is up to date
        if exists(detail_file_path) and getmtime(detail_file_path) > gallery_file_mtime:
            continue

        with open(detail_file_path, "w") as f:
            source_path = abspath(join(app.srcdir, "..", detail['path']))
            f.write(GALLERY_DETAIL.render(filename=detail['name']+'.py', source_path=source_path))

        with open(source_path, "r") as fp:
            code_text = fp.read().replace("show=True", "show=False")
            code_text += EXPORT_CODE.format(filename=join(images_dir, detail["name"]+".png"))
            exec(code_text, {"export_png": export_png})


def setup(app):
    ''' Required Sphinx extension setup function. '''
    app.add_config_value('bokeh_gallery_dir', join("docs", "gallery"), 'html')
    app.connect('config-inited', config_inited_handler)
    app.add_directive('bokeh-gallery', BokehGalleryDirective)

#-----------------------------------------------------------------------------
# Private API
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------
