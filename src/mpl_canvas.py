from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    """
    A canvas for rendering Matplotlib figures in a PyQt application.

    This class integrates Matplotlib with PyQt by extending `FigureCanvas`. It is used
    to draw and display Matplotlib figures within a PyQt widget. The canvas provides
    a subplot and can be customized in terms of its dimensions and resolution. It is
    ideal for embedding visualizations in desktop applications.

    Attributes:
        fig: The Matplotlib Figure object used for rendering plots.
        ax: The subplot of the figure, often used for plotting data.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
