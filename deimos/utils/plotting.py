'''
A minimal set of useful plotting tools

Tom Stuttard
'''

import matplotlib
matplotlib.use('AGG') # In some installations, the 'import matplotlib.pyplot as plt' command below hangs without this (presumanly a issue with a different backend)
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import numpy as np

import os, numbers

def dump_figures_to_pdf(file_path, figures=None):
    '''
    Save figures into a single PDF file
    Can either provide a list of figures, or if not then all in the current session will be saved
    '''

    # Use a list of figures if one is provided by user
    # Otherwise grab all figures from the global list stored by matpotlib
    if figures is not None:
        fig_nums = [
            f.fig.number if isinstance(f, Figure) else f.number for f in figures
        ]  # Handle my Figure class vs matplotligb figure class
    else:
        fig_nums = plt.get_fignums()

    # Check inputs
    file_path = os.path.abspath(file_path)
    parent_dir = os.path.dirname(file_path)
    assert os.path.isdir(parent_dir), 'Could not dump figure to PDF : Directory "%s" does not exist' % parent_dir

    # Check found any figures
    if len(fig_nums) > 0:

        # Create the PDF
        with matplotlib.backends.backend_pdf.PdfPages(file_path) as pdf:

            # Write figures to it
            for fig_num in fig_nums:
                pdf.savefig(fig_num) #TODO option to set dpi

        # Report
        print( "Dumped %i figure(s) to PDF : %s" % (len(fig_nums), file_path) )

    else:
        # Complain
        print("No figures found, could not dump to PDF")


def add_heading_page(text, figsize=None, **kw):
    '''
    Add a headings page to a PDF
    '''

    # Defaults
    if figsize is None :
        figsize = (4,2)

    # Make fig
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.text(0.5, 0.5, text, ha="center", va="center", **kw)


def get_intermediate_points(values, bounding_points=True) :
    '''
    Get the intermediate points between a list of values.
    '''
    
    #TODO replace wit egneric numpy etc function
    #TODO Add log option

    # Checks
    values = np.asarray(values)
    assert len(values.shape) == 1, "`values` must be 1D"

    # Get the delta from the points to the intermediate points
    delta = np.diff(values) / 2.

    # Get the intermediate points
    new_values = values[:-1] + delta

    # Add the bounding points
    if bounding_points :
        new_values = np.concatenate([ [values[0]-delta[0]], new_values, [values[-1]+delta[-1]] ])

    return new_values



def value_spacing_is_linear(values) :
    '''
    Check if an array of values are linearly spaced
    '''
    diff = np.diff(values)
    return np.all(np.isclose(diff, diff[0]))


def plot_colormap(ax, x, y, z, zlabel=None, cbar=True, **kw) :
    '''
    Plot a 2D colormap
    '''

    # Checks
    #TODO

    # Get corners of mesh
    x = get_intermediate_points(x, bounding_points=True)
    y = get_intermediate_points(y, bounding_points=True)
    # assert x.shape[0] == x.size - 2

    # Plot the colormesh
    cmesh = ax.pcolormesh(x, y, z.T, **kw)

    # The mesh is drawn as rectangles. If not drawing edges explicitly, make the edges the same color
    # as the face to avoid an annoying white grid.
    if not any([x in kw for x in ["edgecolor", "edgecolors"]]):
        cmesh.set_edgecolor("face")

    # Add colorbar
    fig = ax.get_figure()
    if cbar :
        cbar = fig.colorbar(cmesh, ax=ax, label=zlabel)



def adjust_lightness(color, amount=0.5):
    '''
    Adjust the lightness of a matplotlib color
    '''
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def get_number_tex(number) :
    '''
    Get a tex representation of a number, handling exponent
    '''

    assert isinstance(number, numbers.Number)

    number_str = "%0.3g" % number 

    # Handle a x 10^b
    if "e" in number_str :
        a, b = number_str.split("e")
        a = float(a)
        b = int(b)
        number_tex = r"$%0.3g \times 10^{%i}$" % (a, b)
    else :
        number_tex = r"$%s$" % number_str

    return number_tex

