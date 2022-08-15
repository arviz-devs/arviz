"""
Sphinx plugin to run example scripts and create a gallery page.

Modified from the seaborn project, which modified the mpld3 project.

Also inspired in bokeh's bokeh_gallery sphinxext.
"""
import glob
import os
import os.path as op
import re
import shutil
import token
import tokenize

from typing import Optional

import matplotlib
import matplotlib.pyplot as plt

from arviz.rcparams import rc_context
from arviz import rcParams

matplotlib.use("Agg")


MPL_RST_TEMPLATE = """
.. _{sphinx_tag}:

{docstring}

.. seealso::

    API Documentation: {api_name}

.. tab-set::
    .. tab-item:: Matplotlib

        .. image:: {img_file}

        .. literalinclude:: {fname}
            :lines: {end_line}-

        .. div:: example-plot-download

           :download:`Download Python Source Code: {fname}<{fname}>`

"""

BOKEH_RST_TEMPLATE = """
    .. tab-item:: Bokeh

        .. bokeh-plot:: {absfname}
            :source-position: none

        .. literalinclude:: {fname}
            :lines: {end_line}-

        .. div:: example-plot-download

           :download:`Download Python Source Code: {fname}<{fname}>`
"""

RST_TEMPLATES = {"matplotlib": MPL_RST_TEMPLATE, "bokeh": BOKEH_RST_TEMPLATE}

INDEX_TEMPLATE = """
:html_theme.sidebar_secondary.remove:

.. _{sphinx_tag}:

Example gallery
===============
{toctree}
{contents}
"""

TOCTREE_START = """
.. toctree::
   :hidden:

"""

CONTENTS_START = """
.. grid:: 1 2 3 3
   :gutter: 2 2 3 3
"""

CONTENTS_ENTRY_TEMPLATE = """
   .. grid-item-card::
      :link: ./{htmlfilename}
      :text-align: center
      :shadow: none
      :class-card: example-gallery

      .. image:: ./matplotlib/{pngfilename}

      +++
      {title}
"""
#:img-top: ./matplotlib/{pngfilename}

# def indent(s, N=3):
#     """Indent a string (Sphinx requires 3)."""
#     return s.replace("\n", "\n" + N * " ")


class ExampleGenerator:
    """Tools for generating an example page from a file"""

    _title: Optional[str]

    def __init__(self, filename, target_dir, backend, target_dir_orig):
        self.filename = filename
        self.target_dir = target_dir
        self.backend = backend
        self._title = None
        self.extract_docstring()
        with open(filename, "r") as fid:
            self.filetext = fid.read()

        outfilename = op.join(target_dir_orig, self.rstfilename)

        # Only actually run it if the output RST file doesn't
        # exist or it was modified less recently than the example
        if not op.exists(outfilename) or (op.getmtime(outfilename) < op.getmtime(filename)):
            self.exec_file()
        else:
            print("skipping {0}".format(self.filename))

    @property
    def title(self) -> str:
        if self._title is not None:
            return self._title
        return self.modulename

    @property
    def dirname(self):
        return op.split(self.filename)[0]

    @property
    def fname(self):
        return op.split(self.filename)[1]

    @property
    def modulename(self) -> str:
        return op.splitext(self.fname)[0]

    @property
    def basename(self) -> str:
        return self.modulename.split("_", 1)[1]

    @property
    def pyfilename(self):
        return self.modulename + ".py"

    @property
    def rstfilename(self):
        return self.basename + ".rst"

    @property
    def htmlfilename(self):
        return self.basename + ".html"

    @property
    def pngfilename(self):
        pngfile = self.modulename + ".png"
        return "_images/" + pngfile

    @property
    def apiname(self):
        with open(op.join(self.target_dir, self.pyfilename), "r") as file:
            regex = r"az\.(plot\_[a-z_]+)\("
            name = re.findall(regex, file.read())
        apitext = name[0] if name else ""
        return (
            ":func:`~arviz.{apitext}`".format(apitext=apitext)
            if apitext
            else "No API Documentation available"
        )

    @property
    def sphinxtag(self):
        return f"example_{self.basename}"

    @property
    def pagetitle(self):
        return self.docstring.strip().split("\n")[0].strip()

    def extract_docstring(self):
        """Extract a module-level docstring"""
        lines = open(self.filename).readlines()
        start_row = 0
        if lines[0].startswith("#!"):
            lines.pop(0)
            start_row = 1

        docstring = ""
        first_par = ""
        line_iter = lines.__iter__()
        tokens = tokenize.generate_tokens(lambda: next(line_iter))
        for tok_type, tok_content, _, (erow, _), _ in tokens:
            tok_type = token.tok_name[tok_type]
            if tok_type in ("NEWLINE", "COMMENT", "NL", "INDENT", "DEDENT"):
                continue
            elif tok_type == "STRING":
                docstring = eval(tok_content)
                # If the docstring is formatted with several paragraphs,
                # extract the first one:
                paragraphs = "\n".join(line.rstrip() for line in docstring.split("\n")).split(
                    "\n\n"
                )
                if len(paragraphs) > 0:
                    first_par = paragraphs[0]
            break

        # thumbloc = None
        # title: Optional[str] = None
        ex_title: str = ""
        for line in docstring.split("\n"):
            # # we've found everything we need...
            # if thumbloc and title and ex_title != "":
            #     break
            # m = re.match(r"^_thumb: (\.\d+),\s*(\.\d+)", line)
            # if m:
            #     thumbloc = float(m.group(1)), float(m.group(2))
            #     continue
            # m = re.match(r"^_example_title: (.*)$", line)
            # if m:
            #     title = m.group(1)
            #     continue
            # capture the first non-empty line of the docstring as title
            if ex_title == "":
                ex_title = line
        assert ex_title != ""
        # if thumbloc is not None:
        #     self.thumbloc = thumbloc
        #     docstring = "\n".join([l for l in docstring.split("\n") if not l.startswith("_thumb")])

        self._title = ex_title

        self.docstring = docstring
        self.short_desc = first_par
        self.end_line = erow + 1 + start_row  # pylint: disable=undefined-loop-variable

    def exec_file(self):
        # pylint: disable=exec-used
        print("running {0}".format(self.filename))

        plt.close("all")
        if self.backend == "matplotlib":
            pngfile = op.join(self.target_dir, self.pngfilename)
            my_globals = {"plt": plt}
            with open(self.filename, "r") as fp:
                code_text = fp.read()
                code_text = re.sub(r"(plt\.show\S+)", "", code_text)
                exec(compile(code_text, self.filename, "exec"), my_globals)

            fig = plt.gcf()
            fig.canvas.draw()
            fig.savefig(pngfile, dpi=75)


    def toctree_entry(self):
        return "   {}\n".format(op.join(op.splitext(self.htmlfilename)[0]))

    def contents_entry(self) -> str:
        return CONTENTS_ENTRY_TEMPLATE.format(
            htmlfilename=self.htmlfilename,
            pngfilename=self.pngfilename,
            sphinx_tag=self.sphinxtag,
            title=self.title,
        )

def main(app):
    # Get paths for files
    working_dir = os.getcwd()
    os.chdir(app.builder.srcdir)
    static_dir = op.join(app.builder.srcdir, "..", "build", "_static")
    target_dir_orig = op.join(app.builder.srcdir, "examples")

    if not op.exists(static_dir):
        os.makedirs(static_dir)

    # Map each backend to their respective paths
    path_dict = {}
    backends = ("matplotlib", "bokeh")
    backend_prefixes = ("mpl", "bokeh")
    for backend in backends:
        target_dir = op.join(target_dir_orig, backend)
        image_dir = op.join(target_dir, "_images")
        source_dir = op.abspath(op.join(app.builder.srcdir, "..", "..", "examples", backend))

        if not op.exists(source_dir):
            os.makedirs(source_dir)

        if not op.exists(target_dir):
            os.makedirs(target_dir)

        if not op.exists(image_dir):
            os.makedirs(image_dir)

        path_dict[backend] = {
            "source_dir": source_dir,
            "target_dir": target_dir,
            "image_dir": image_dir,
        }

    # Begin templates for table of contents and content cards
    toctree = TOCTREE_START
    contents = CONTENTS_START

    # Write individual example files
    files = sorted(glob.glob(op.join(path_dict["matplotlib"]["source_dir"], "*.py")))
    for filename in files:
        base_filename = op.split(filename)[1].split("_", 1)[1]
        example_contents = ""
        for backend, prefix in zip(backends, backend_prefixes):
            source_dir = path_dict[backend]["source_dir"]
            target_dir = path_dict[backend]["target_dir"]
            expected_filename = op.join(source_dir, f"{prefix}_{base_filename}")

            if not op.exists(expected_filename):
                if backend == "matplotlib":
                    raise ValueError("All examples must have a matplotlib counterpart.")
                continue

            ex = ExampleGenerator(
                expected_filename, target_dir, backend, target_dir_orig
            )

            shutil.copyfile(expected_filename, op.join(target_dir, ex.pyfilename))
            output = RST_TEMPLATES[backend].format(
                sphinx_tag=ex.sphinxtag,
                docstring=ex.docstring,
                end_line=ex.end_line,
                fname=op.join(backend, ex.pyfilename),
                absfname=op.join(target_dir, ex.pyfilename),
                img_file=op.join(backend, ex.pngfilename),
                api_name=ex.apiname,
            )
            example_contents += output

            # Add plot to table of contents and content card if matplotlib
            if backend == "matplotlib":
                toctree += ex.toctree_entry()
                contents += ex.contents_entry()

        with open(op.join(target_dir_orig, ex.rstfilename), "w") as f:
            f.write(example_contents)


    # Write index file
    index_file = op.join(target_dir, "..", "index.rst")
    with open(index_file, "w") as index:
        index.write(
            INDEX_TEMPLATE.format(
                sphinx_tag="example_gallery",
                toctree=toctree,
                contents=contents,
                examples_source=source_dir,
            )
        )

    os.chdir(working_dir)


def setup(app):
    app.connect("builder-inited", main)
