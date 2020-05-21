"""Matplotib Posterior predictive plot."""
import platform
import logging
from matplotlib import animation, get_backend
import matplotlib.pyplot as plt
import numpy as np

from arviz.plots.backends.matplotlib import backend_show
from arviz.plots.kdeplot import plot_kde
from arviz.plots.plot_utils import (
    make_label,
    _create_axes_grid,
)
from arviz.numeric_utils import _fast_kde, histogram, get_bins

_log = logging.getLogger(__name__)


def plot_ppc_mpl(
    ax,
    length_plotters,
    rows,
    cols,
    figsize,
    animated,
    obs_plotters,
    pp_plotters,
    predictive_dataset,
    pp_sample_ix,
    kind,
    alpha,
    linewidth,
    mean,
    xt_labelsize,
    ax_labelsize,
    jitter,
    total_pp_samples,
    legend,
    group,
    markersize,
    animation_kwargs,
    num_pp_samples,
    backend_kwargs,
    show,
):
    """Matplotlib ppc plot."""
    if animated:
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell" and get_backend() != "nbAgg":
                raise Warning(
                    "To run animations inside a notebook you have to use the nbAgg backend. "
                    "Try with `%matplotlib notebook` or  `%matplotlib  nbAgg`. You can switch "
                    "back to the default backend with `%matplotlib  inline` or "
                    "`%matplotlib  auto`."
                )
        except NameError:
            pass

        if animation_kwargs["blit"] and platform.system() != "Linux":
            _log.warning(
                "If you experience problems rendering the animation try setting "
                "`animation_kwargs({'blit':False}) or changing the plotting backend "
                "(e.g. to TkAgg)"
            )

    if ax is None:
        fig, axes = _create_axes_grid(
            length_plotters, rows, cols, figsize=figsize, backend_kwargs=backend_kwargs
        )
    else:
        axes = np.ravel(ax)
        if len(axes) != length_plotters:
            raise ValueError(
                "Found {} variables to plot but {} axes instances. They must be equal.".format(
                    length_plotters, len(axes)
                )
            )
        if animated:
            fig = axes[0].get_figure()
            if not all([ax.get_figure() is fig for ax in axes]):
                raise ValueError("All axes must be on the same figure for animation to work")

    for i, ax_i in enumerate(axes):
        var_name, selection, obs_vals = obs_plotters[i]
        pp_var_name, _, pp_vals = pp_plotters[i]
        dtype = predictive_dataset[pp_var_name].dtype.kind

        # flatten non-specified dimensions
        obs_vals = obs_vals.flatten()
        pp_vals = pp_vals.reshape(total_pp_samples, -1)
        pp_sampled_vals = pp_vals[pp_sample_ix]
        # This logic doesn't work
        # if after flattening accross chains and draws there are
        # no additional axis.
        if pp_sampled_vals.shape[-1] == 1:
            pp_sampled_vals = pp_sampled_vals.T
        if kind == "kde":
            plot_kwargs = {"color": "C5", "alpha": alpha, "linewidth": 0.5 * linewidth}
            if dtype == "i":
                plot_kwargs["drawstyle"] = "steps-pre"
            ax_i.plot(
                [], color="C5", label="{} predictive {}".format(group.capitalize(), pp_var_name)
            )

            if dtype == "f":
                plot_kde(
                    obs_vals,
                    label="Observed {}".format(var_name),
                    plot_kwargs={"color": "k", "linewidth": linewidth, "zorder": 3},
                    fill_kwargs={"alpha": 0},
                    ax=ax_i,
                    legend=legend,
                )
            else:
                bins = get_bins(obs_vals)
                _, hist, bin_edges = histogram(obs_vals, bins=bins)
                hist = np.concatenate((hist[:1], hist))
                ax_i.plot(
                    bin_edges,
                    hist,
                    label="Observed {}".format(var_name),
                    color="k",
                    linewidth=linewidth,
                    zorder=3,
                    drawstyle=plot_kwargs["drawstyle"],
                )

            pp_densities = []
            pp_xs = []
            for vals in pp_sampled_vals:
                vals = np.array([vals]).flatten()
                if dtype == "f":
                    pp_density, lower, upper = _fast_kde(vals)
                    pp_x = np.linspace(lower, upper, len(pp_density))
                    pp_densities.append(pp_density)
                    pp_xs.append(pp_x)
                else:
                    bins = get_bins(vals)
                    _, hist, bin_edges = histogram(vals, bins=bins)
                    hist = np.concatenate((hist[:1], hist))
                    pp_densities.append(hist)
                    pp_xs.append(bin_edges)

            if animated:
                animate, init = _set_animation(
                    pp_sampled_vals, ax_i, dtype=dtype, kind=kind, plot_kwargs=plot_kwargs
                )

            else:
                if dtype == "f":
                    ax_i.plot(np.transpose(pp_xs), np.transpose(pp_densities), **plot_kwargs)
                else:
                    for x_s, y_s in zip(pp_xs, pp_densities):
                        ax_i.plot(x_s, y_s, **plot_kwargs)

            if mean:
                label = "{} predictive mean {}".format(group.capitalize(), pp_var_name)
                if dtype == "f":
                    rep = len(pp_densities)
                    len_density = len(pp_densities[0])
                    new_x = np.linspace(np.min(pp_xs), np.max(pp_xs), len_density)
                    new_d = np.zeros((rep, len_density))
                    bins = np.digitize(pp_xs, new_x, right=True)
                    new_x -= (new_x[1] - new_x[0]) / 2
                    for irep in range(rep):
                        new_d[irep][bins[irep]] = pp_densities[irep]
                    ax_i.plot(
                        new_x,
                        new_d.mean(0),
                        color="C0",
                        linestyle="--",
                        linewidth=linewidth,
                        zorder=2,
                        label=label,
                    )
                else:
                    vals = pp_vals.flatten()
                    bins = get_bins(vals)
                    _, hist, bin_edges = histogram(vals, bins=bins)
                    hist = np.concatenate((hist[:1], hist))
                    ax_i.plot(
                        bin_edges,
                        hist,
                        color="C0",
                        linewidth=linewidth,
                        label=label,
                        zorder=2,
                        linestyle="--",
                        drawstyle=plot_kwargs["drawstyle"],
                    )
            ax_i.tick_params(labelsize=xt_labelsize)
            ax_i.set_yticks([])

        elif kind == "cumulative":
            drawstyle = "default" if dtype == "f" else "steps-pre"
            ax_i.plot(
                *_empirical_cdf(obs_vals),
                color="k",
                linewidth=linewidth,
                label="Observed {}".format(var_name),
                drawstyle=drawstyle,
                zorder=3
            )
            if animated:
                animate, init = _set_animation(
                    pp_sampled_vals,
                    ax_i,
                    kind=kind,
                    alpha=alpha,
                    drawstyle=drawstyle,
                    linewidth=linewidth,
                )

            else:
                pp_densities = np.empty((2 * len(pp_sampled_vals), pp_sampled_vals[0].size))
                for idx, vals in enumerate(pp_sampled_vals):
                    vals = np.array([vals]).flatten()
                    pp_x, pp_density = _empirical_cdf(vals)
                    pp_densities[2 * idx] = pp_x
                    pp_densities[2 * idx + 1] = pp_density

                ax_i.plot(
                    *pp_densities, alpha=alpha, color="C5", drawstyle=drawstyle, linewidth=linewidth
                )
            ax_i.plot([], color="C5", label="Posterior predictive {}".format(pp_var_name))
            if mean:
                ax_i.plot(
                    *_empirical_cdf(pp_vals.flatten()),
                    color="C0",
                    linestyle="--",
                    linewidth=linewidth,
                    drawstyle=drawstyle,
                    label="Posterior predictive mean {}".format(pp_var_name)
                )
            ax_i.set_yticks([0, 0.5, 1])

        elif kind == "scatter":
            if mean:
                if dtype == "f":
                    plot_kde(
                        pp_vals.flatten(),
                        plot_kwargs={
                            "color": "C0",
                            "linestyle": "--",
                            "linewidth": linewidth,
                            "zorder": 3,
                        },
                        label="Posterior predictive mean {}".format(pp_var_name),
                        ax=ax_i,
                        legend=legend,
                    )
                else:
                    vals = pp_vals.flatten()
                    bins = get_bins(vals)
                    _, hist, bin_edges = histogram(vals, bins=bins)
                    hist = np.concatenate((hist[:1], hist))
                    ax_i.plot(
                        bin_edges,
                        hist,
                        color="C0",
                        linewidth=linewidth,
                        label="Posterior predictive mean {}".format(pp_var_name),
                        zorder=3,
                        linestyle="--",
                        drawstyle="steps-pre",
                    )

            _, limit = ax_i.get_ylim()
            limit *= 1.05
            y_rows = np.linspace(0, limit, num_pp_samples + 1)
            jitter_scale = y_rows[1] - y_rows[0]
            scale_low = 0
            scale_high = jitter_scale * jitter

            obs_yvals = np.zeros_like(obs_vals, dtype=np.float64)
            if jitter:
                obs_yvals += np.random.uniform(low=scale_low, high=scale_high, size=len(obs_vals))
            ax_i.plot(
                obs_vals,
                obs_yvals,
                "o",
                color="C0",
                markersize=markersize,
                alpha=alpha,
                label="Observed {}".format(var_name),
                zorder=4,
            )

            if animated:
                animate, init = _set_animation(
                    pp_sampled_vals,
                    ax_i,
                    kind=kind,
                    height=y_rows.mean() * 0.5,
                    markersize=markersize,
                )

            else:
                for vals, y in zip(pp_sampled_vals, y_rows[1:]):
                    vals = np.ravel(vals)
                    yvals = np.full_like(vals, y, dtype=np.float64)
                    if jitter:
                        yvals += np.random.uniform(low=scale_low, high=scale_high, size=len(vals))
                    ax_i.plot(
                        vals, yvals, "o", zorder=2, color="C5", markersize=markersize, alpha=alpha
                    )

            ax_i.plot([], "C5o", label="Posterior predictive {}".format(pp_var_name))

            ax_i.set_yticks([])

        if var_name != pp_var_name:
            xlabel = "{} / {}".format(var_name, pp_var_name)
        else:
            xlabel = var_name
        ax_i.set_xlabel(make_label(xlabel, selection), fontsize=ax_labelsize)

        if legend:
            if i == 0:
                ax_i.legend(fontsize=xt_labelsize * 0.75)
            else:
                ax_i.legend([])

    if backend_show(show):
        plt.show()

    if animated:
        ani = animation.FuncAnimation(
            fig, animate, np.arange(0, num_pp_samples), init_func=init, **animation_kwargs
        )
        return axes, ani
    else:
        return axes


def _set_animation(
    pp_sampled_vals,
    ax,
    dtype=None,
    kind="density",
    alpha=None,
    drawstyle=None,
    linewidth=None,
    height=None,
    markersize=None,
    plot_kwargs=None,
):
    if kind == "kde":
        length = len(pp_sampled_vals)
        if dtype == "f":
            y_vals, lower, upper = _fast_kde(pp_sampled_vals[0])
            x_vals = np.linspace(lower, upper, len(y_vals))

            max_max = max([max(_fast_kde(pp_sampled_vals[i])[0]) for i in range(length)])

            ax.set_ylim(0, max_max)

            (line,) = ax.plot(x_vals, y_vals, **plot_kwargs)

            def animate(i):
                y_vals, lower, upper = _fast_kde(pp_sampled_vals[i])
                x_vals = np.linspace(lower, upper, len(y_vals))
                line.set_data(x_vals, y_vals)
                return line

        else:
            vals = pp_sampled_vals[0]
            _, y_vals, x_vals = histogram(vals, bins="auto")
            (line,) = ax.plot(x_vals[:-1], y_vals, **plot_kwargs)

            max_max = max(
                [max(histogram(pp_sampled_vals[i], bins="auto")[1]) for i in range(length)]
            )

            ax.set_ylim(0, max_max)

            def animate(i):
                _, y_vals, x_vals = histogram(pp_sampled_vals[i], bins="auto")
                line.set_data(x_vals[:-1], y_vals)
                return (line,)

    elif kind == "cumulative":
        x_vals, y_vals = _empirical_cdf(pp_sampled_vals[0])
        (line,) = ax.plot(
            x_vals, y_vals, alpha=alpha, color="C5", drawstyle=drawstyle, linewidth=linewidth
        )

        def animate(i):
            x_vals, y_vals = _empirical_cdf(pp_sampled_vals[i])
            line.set_data(x_vals, y_vals)
            return line

    elif kind == "scatter":
        x_vals = pp_sampled_vals[0]
        y_vals = np.full_like(x_vals, height, dtype=np.float64)
        (line,) = ax.plot(
            x_vals, y_vals, "o", zorder=2, color="C5", markersize=markersize, alpha=alpha
        )

        def animate(i):
            line.set_xdata(np.ravel(pp_sampled_vals[i]))
            return line

    def init():
        if kind != "scatter":
            line.set_data([], [])
        else:
            line.set_xdata([])
        return line

    return animate, init


def _empirical_cdf(data):
    """Compute empirical cdf of a numpy array.
    Parameters
    ----------
    data : np.array
        1d array
    Returns
    -------
    np.array, np.array
        x and y coordinates for the empirical cdf of the data
    """
    return np.sort(data), np.linspace(0, 1, len(data))


"""Posterior/Prior predictive plot."""
from numbers import Integral
import platform
import logging
import numpy as np

from arviz.plots.plot_utils import (
    xarray_var_iter,
    _scale_fig_size,
    default_grid,
    filter_plotters_list,
    get_plotting_function,
)
from arviz.rcparams import rcParams
from arviz.utils import _var_names

_log = logging.getLogger(__name__)


def plot_ppc(
    data,
    kind="kde",
    alpha=None,
    mean=True,
    figsize=None,
    textsize=None,
    data_pairs=None,
    var_names=None,
    filter_vars=None,
    coords=None,
    flatten=None,
    flatten_pp=None,
    num_pp_samples=None,
    random_seed=None,
    jitter=None,
    animated=False,
    animation_kwargs=None,
    legend=True,
    ax=None,
    backend=None,
    backend_kwargs=None,
    group="posterior",
    show=None,
):
    """
    Plot for posterior/prior predictive checks.
    Parameters
    ----------
    data: az.InferenceData object
        InferenceData object containing the observed and posterior/prior predictive data.
    kind: str
        Type of plot to display (kde, cumulative, or scatter). Defaults to kde.
    alpha: float
        Opacity of posterior/prior predictive density curves.
        Defaults to 0.2 for kind = kde and cumulative, for scatter defaults to 0.7
    mean: bool
        Whether or not to plot the mean posterior/prior predictive distribution. Defaults to True
    figsize: tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be
        autoscaled based on figsize.
    data_pairs: dict
        Dictionary containing relations between observed data and posterior/prior predictive data.
        Dictionary structure:
        - key = data var_name
        - value = posterior/prior predictive var_name
        For example, `data_pairs = {'y' : 'y_hat'}`
        If None, it will assume that the observed data and the posterior/prior
        predictive data have the same variable name.
    var_names: list of variable names
        Variables to be plotted, if `None` all variable are plotted. Prefix the
        variables by `~` when you want to exclude them from the plot.
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.
    coords: dict
        Dictionary mapping dimensions to selected coordinates to be plotted.
        Dimensions without a mapping specified will include all coordinates for
        that dimension. Defaults to including all coordinates for all
        dimensions if None.
    flatten: list
        List of dimensions to flatten in observed_data. Only flattens across the coordinates
        specified in the coords argument. Defaults to flattening all of the dimensions.
    flatten_pp: list
        List of dimensions to flatten in posterior_predictive/prior_predictive. Only flattens
        across the coordinates specified in the coords argument. Defaults to flattening all
        of the dimensions. Dimensions should match flatten excluding dimensions for data_pairs
        parameters. If flatten is defined and flatten_pp is None, then `flatten_pp=flatten`.
    num_pp_samples: int
        The number of posterior/prior predictive samples to plot. For `kind` = 'scatter' and
        `animation = False` if defaults to a maximum of 5 samples and will set jitter to 0.7
        unless defined otherwise. Otherwise it defaults to all provided samples.
    random_seed: int
        Random number generator seed passed to numpy.random.seed to allow
        reproducibility of the plot. By default, no seed will be provided
        and the plot will change each call if a random sample is specified
        by `num_pp_samples`.
    jitter: float
        If kind is "scatter", jitter will add random uniform noise to the height
        of the ppc samples and observed data. By default 0.
    animated: bool
        Create an animation of one posterior/prior predictive sample per frame. Defaults to False.
        Only works with matploblib backend.
        To run animations inside a notebook you have to use the `nbAgg` matplotlib's backend.
        Try with `%matplotlib notebook` or  `%matplotlib  nbAgg`. You can switch back to the
        default matplotlib's backend with `%matplotlib  inline` or `%matplotlib  auto`.
        If switching back and forth between matplotlib's backend, you may need to run twice the cell
        with the animation.
        If you experience problems rendering the animation try setting
        `animation_kwargs({'blit':False}) or changing the matplotlib's backend (e.g. to TkAgg)
        If you run the animation from a script write `ax, ani = az.plot_ppc(.)`
    animation_kwargs : dict
        Keywords passed to `animation.FuncAnimation`. Ignored with matploblib backend.
    legend : bool
        Add legend to figure. By default True.
    ax: numpy array-like of matplotlib axes or bokeh figures, optional
        A 2D array of locations into which to plot the densities. If not supplied, Arviz will create
        its own array of plot areas (and return it).
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    group: {"prior", "posterior"}, optional
        Specifies which InferenceData group should be plotted. Defaults to 'posterior'.
        Other value can be 'prior'.
    show: bool, optional
        Call backend show function.
    Returns
    -------
    axes: matplotlib axes or bokeh figures
    Examples
    --------
    Plot the observed data KDE overlaid on posterior predictive KDEs.
    .. plot::
        :context: close-figs
        >>> import arviz as az
        >>> data = az.load_arviz_data('radon')
        >>> az.plot_ppc(data, data_pairs={"obs":"obs"})
        >>> #az.plot_ppc(data, data_pairs={"obs":"obs_hat"})
    Plot the overlay with empirical CDFs.
    .. plot::
        :context: close-figs
        >>> az.plot_ppc(data, kind='cumulative')
    Use the coords and flatten parameters to plot selected variable dimensions
    across multiple plots.
    .. plot::
        :context: close-figs
        >>> az.plot_ppc(data, coords={'observed_county': ['ANOKA', 'BELTRAMI']}, flatten=[])
    Plot the overlay using a stacked scatter plot that is particularly useful
    when the sample sizes are small.
    .. plot::
        :context: close-figs
        >>> az.plot_ppc(data, kind='scatter', flatten=[],
        >>>             coords={'observed_county': ['AITKIN', 'BELTRAMI']})
    Plot random posterior predictive sub-samples.
    .. plot::
        :context: close-figs
        >>> az.plot_ppc(data, num_pp_samples=30, random_seed=7)
    """
    if group not in ("posterior", "prior"):
        raise TypeError("`group` argument must be either `posterior` or `prior`")

    for groups in ("{}_predictive".format(group), "observed_data"):
        if not hasattr(data, groups):
            raise TypeError(
                '`data` argument must have the group "{group}" for ppcplot'.format(group=groups)
            )

    if kind.lower() not in ("kde", "cumulative", "scatter"):
        raise TypeError("`kind` argument must be either `kde`, `cumulative`, or `scatter`")

    if data_pairs is None:
        data_pairs = {}

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    if animation_kwargs is None:
        animation_kwargs = {}
    if platform.system() == "Linux":
        animation_kwargs.setdefault("blit", True)
    else:
        animation_kwargs.setdefault("blit", False)

    if alpha is None:
        if animated:
            alpha = 1
        else:
            if kind.lower() == "scatter":
                alpha = 0.7
            else:
                alpha = 0.2

    if jitter is None:
        jitter = 0.0
    assert jitter >= 0.0

    observed = data.observed_data

    if group == "posterior":
        predictive_dataset = data.posterior_predictive
    elif group == "prior":
        predictive_dataset = data.prior_predictive

    if var_names is None:
        var_names = list(observed.data_vars)
    var_names = _var_names(var_names, observed, filter_vars)
    pp_var_names = [data_pairs.get(var, var) for var in var_names]
    pp_var_names = _var_names(pp_var_names, predictive_dataset, filter_vars)

    if flatten_pp is None and flatten is None:
        flatten_pp = list(predictive_dataset.dims.keys())
    elif flatten_pp is None:
        flatten_pp = flatten
    if flatten is None:
        flatten = list(observed.dims.keys())

    if coords is None:
        coords = {}

    if random_seed is not None:
        np.random.seed(random_seed)

    total_pp_samples = predictive_dataset.sizes["chain"] * predictive_dataset.sizes["draw"]
    if num_pp_samples is None:
        if kind == "scatter" and not animated:
            num_pp_samples = min(5, total_pp_samples)
        else:
            num_pp_samples = total_pp_samples

    if (
        not isinstance(num_pp_samples, Integral)
        or num_pp_samples < 1
        or num_pp_samples > total_pp_samples
    ):
        raise TypeError(
            "`num_pp_samples` must be an integer between 1 and "
            + "{limit}.".format(limit=total_pp_samples)
        )

    pp_sample_ix = np.random.choice(total_pp_samples, size=num_pp_samples, replace=False)

    for key in coords.keys():
        coords[key] = np.where(np.in1d(observed[key], coords[key]))[0]

    obs_plotters = filter_plotters_list(
        list(
            xarray_var_iter(
                observed.isel(coords), skip_dims=set(flatten), var_names=var_names, combined=True
            )
        ),
        "plot_ppc",
    )
    length_plotters = len(obs_plotters)
    pp_plotters = [
        tup
        for _, tup in zip(
            range(length_plotters),
            xarray_var_iter(
                predictive_dataset.isel(coords),
                var_names=pp_var_names,
                skip_dims=set(flatten_pp),
                combined=True,
            ),
        )
    ]
    rows, cols = default_grid(length_plotters)

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    ppcplot_kwargs = dict(
        ax=ax,
        length_plotters=length_plotters,
        rows=rows,
        cols=cols,
        figsize=figsize,
        animated=animated,
        obs_plotters=obs_plotters,
        pp_plotters=pp_plotters,
        predictive_dataset=predictive_dataset,
        pp_sample_ix=pp_sample_ix,
        kind=kind,
        alpha=alpha,
        linewidth=linewidth,
        mean=mean,
        xt_labelsize=xt_labelsize,
        ax_labelsize=ax_labelsize,
        jitter=jitter,
        total_pp_samples=total_pp_samples,
        legend=legend,
        group=group,
        markersize=markersize,
        animation_kwargs=animation_kwargs,
        num_pp_samples=num_pp_samples,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend == "bokeh":
        if animated:
            raise TypeError("Animation option is only supported with matplotlib backend.")

        ppcplot_kwargs.pop("animated")
        ppcplot_kwargs.pop("animation_kwargs")
        ppcplot_kwargs.pop("legend")
        ppcplot_kwargs.pop("group")
        ppcplot_kwargs.pop("xt_labelsize")
        ppcplot_kwargs.pop("ax_labelsize")

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_ppc", "ppcplot", backend)
    axes = plot_ppc_mpl(**ppcplot_kwargs)
    return axes