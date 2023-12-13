# system import
import io

# third party import
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# local import


def plot_3d_figure(table: pd.DataFrame):
    """Create a matplotlib 3d figure, //export and save in log
    table: pd.DataFrame
    """

    df = table.unstack().reset_index()
    df.columns = pd.Index(["throttle", "speed", "torque"])

    fig = plt.figure(visible=False)
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_trisurf(  # type: ignore
        df["throttle"],
        df["speed"],
        df["torque"],
        cmap=plt.get_cmap("YlGnBu"),
        linewidth=5,
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=30, azim=175)  # type: ignore

    return fig
    # plt.show()
    # time.sleep(5)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
