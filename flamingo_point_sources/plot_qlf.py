#!/bin/env python
"""
Plot the quasar luminosity function (QLF) of the abundance matched black holes against the Shen+2020 best fitting QLF 
model they were sampled from. 

For a QLF model at a given redshift, sample the QLF to assign the haloes in the overlapping redshift range 
new bolometric luminosities. The assigned luminosities are binned into a QLF:

    phi(L) = N(L) / (V dlogL)        [Mpc^-3 dex^-1]

with V the comoving volume the luminosities were sampled for.

The top panel of the output figure shows phi with Poisson errors and the best fitting QLF model from Shen+2020. 
The bottom panel shows the ratio of the two curves, abundance matched / model
With --per_shell a second figure shows the QLF of the haloes in each lightcone shell of the range, each with the volume of its shell.

The luminosities assigned are a permutation of the same sample whatever the ranking (sigma_Y), so the QLF of the
whole redshift range does not depend on sigma_Y; the per-shell QLFs do, since the ranking decides which shell's haloes
get the bright luminosities.

Run from the top of the repository, e.g.
    python -m flamingo_point_sources.plot_abundance_matched_qlf --box_res L1000N1800 --sim_name HYDRO_FIDUCIAL \\
        --qlf_shell_nr 5 --qlf_filename QLF_models/Shen2020_local_polished.dat --sigma_Y -1 \\
        --output plots/abundance_matched_qlf_row5.png --per_shell
"""
import os
import argparse
import numpy as np
import unyt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

from .sample_qlf import SampleQLF


# build plot settings

def plot_settings():
    """
    matplotlib settings for this example. 
    """
    import matplotlib as mpl
    
    # Line Properties
    plt.rcParams["lines.linewidth"]=1.5 
    # Font options
    plt.rcParams["font.size"]=8
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["text.usetex"] = False
    plt.rcParams["legend.labelspacing"]=0.25
    plt.rcParams["legend.columnspacing"]=0.75
    plt.rcParams["legend.borderpad"]=0.3
    plt.rcParams["legend.borderaxespad"]=0.7
    plt.rcParams["legend.fontsize"]=10
    # Figure layour settings
    plt.rcParams["figure.constrained_layout.use"] =True
    plt.rcParams["figure.constrained_layout.h_pad"] =0.005
    plt.rcParams["figure.constrained_layout.w_pad"] =0.005
    plt.rcParams["figure.constrained_layout.hspace"]=0.005
    plt.rcParams["figure.constrained_layout.wspace"]=0.005
    # axes font settings
    mpl.rcParams['axes.labelsize']=10
    mpl.rcParams['figure.labelsize']=10
    mpl.rcParams['xtick.labelsize']=10
    mpl.rcParams['ytick.labelsize']=10
    # axes tick settings
    plt.rcParams["xtick.direction"] ='in'
    plt.rcParams["ytick.direction"] ='in'
    plt.rcParams["xtick.top"]=True
    plt.rcParams["ytick.right"]= True

# define plotting colours
model_colour = "k" #"#fe9c46"
label_colour = "#52514e"
abundance_matched_colour = "#4682B4"
shell_cmap = LinearSegmentedColormap.from_list("magenta_cyan_gold", ["#8b008b", "#008c99", "#b8860b"])


def luminosity_bin_edges(log_Lmin, log_L_faintest, log_Lmax, dlogL):
    """
    Returns regular log10 bolometric luminosity bin edges [erg/s] on the grid log_Lmin + k dlogL, from the first edge
    at or above the faintest luminosity, covering log_Lmax.
    Starting at the faintest luminosity leaves out the partly filled bin at the faint end: the sampler stops a little
    above Log_Lmin (its last 0.1 dex bin above Log_Lmin is not drawn), or where it runs out of haloes.

    :param  log_Lmin:       origin of the bin grid, log10 bolometric luminosity [erg/s]
    :type   log_Lmin:       float
    :param  log_L_faintest: faintest log10 luminosity [erg/s] in the sample
    :type   log_L_faintest: float
    :param  log_Lmax:       luminosity the last bin must reach, log10 [erg/s]
    :type   log_Lmax:       float
    :param  dlogL:          width of the bins in log10 luminosity
    :type   dlogL:          float
    """
    first = log_Lmin + dlogL * np.ceil((log_L_faintest - log_Lmin) / dlogL - 1e-9)
    n_bins = max(1, int(np.ceil((log_Lmax - first) / dlogL)))
    return first + dlogL * np.arange(n_bins + 1)


def binned_qlf(Lbol, volume, log_L_edges):
    """
    The QLF of a set of bolometric luminosities: the number in each log10 luminosity bin divided by the volume and the
    bin width. Luminosities <= 0 (haloes without an AGN) are not counted.

    Returns a dictionary of np.ndarray:
        'log_L_centres', 'counts', 'phi' and its Poisson error 'phi_err' [Mpc^-3 dex^-1]

    :param  Lbol:           bolometric luminosities, erg/s if they have no units
    :type   Lbol:           unyt.unyt_array or np.ndarray
    :param  volume:         comoving volume the luminosities are in, Mpc^3 if it has no units
    :type   volume:         unyt.unyt_quantity or float
    :param  log_L_edges:    regular log10 bolometric luminosity bin edges [erg/s]
    :type   log_L_edges:    np.ndarray
    """
    if hasattr(Lbol, "units"):
        Lbol = Lbol.to_value(unyt.erg / unyt.s)
    if hasattr(volume, "units"):
        volume = volume.to_value(unyt.Mpc**3)
    Lbol = np.asarray(Lbol, dtype=np.float64)
    log_L = np.log10(Lbol[Lbol > 0])

    counts, _ = np.histogram(log_L, bins=log_L_edges)
    dlogL = np.diff(log_L_edges)
    norm = 1. / (float(volume) * dlogL)
    return {
        "log_L_centres": 0.5 * (log_L_edges[1:] + log_L_edges[:-1]),
        "counts": counts,
        "phi": counts * norm,
        "phi_err": np.sqrt(counts) * norm,
    }


def model_qlf(qlf_filename, qlf_redshift, log_L):
    """
    The Shen+2020 QLF at the tabulated redshift nearest qlf_redshift.

    Returns the QLF [Mpc^-3 dex^-1] at each log10 bolometric luminosity, as an np.ndarray.

    :param  qlf_filename:   QLF parameter file, see SampleQLF
    :type   qlf_filename:   str
    :param  qlf_redshift:   redshift selecting the QLF
    :type   qlf_redshift:   float
    :param  log_L:          log10 bolometric luminosities [erg/s]
    :type   log_L:          np.ndarray
    """
    qlf = SampleQLF(qlf_filename)
    qlf.update_qlf_redshift(qlf_redshift)
    return qlf.qlf_values(unyt.unyt_array(10**np.asarray(log_L, dtype=np.float64), units=unyt.erg / unyt.s))


def model_qlf_bin_averages(qlf_filename, qlf_redshift, log_L_edges, points_per_bin=64):
    """
    The Shen+2020 QLF averaged over each log10 luminosity bin, to compare with a binned QLF: the QLF is curved, so its
    value at a bin centre differs from its bin average.

    Returns the average QLF in each bin [Mpc^-3 dex^-1], as an np.ndarray.

    :param  qlf_filename:   QLF parameter file, see SampleQLF
    :type   qlf_filename:   str
    :param  qlf_redshift:   redshift selecting the QLF
    :type   qlf_redshift:   float
    :param  log_L_edges:    log10 bolometric luminosity bin edges [erg/s]
    :type   log_L_edges:    np.ndarray
    :param  points_per_bin: number of points the QLF is averaged over in each bin (midpoint rule)
    :type   points_per_bin: int
    """
    fractions = (np.arange(points_per_bin) + 0.5) / points_per_bin
    log_L = log_L_edges[:-1, None] + np.diff(log_L_edges)[:, None] * fractions[None, :]
    return model_qlf(qlf_filename, qlf_redshift, log_L.ravel()).reshape(log_L.shape).mean(axis=1)




def plot_qlf_comparison(binned, log_L_model, phi_model, phi_model_in_bins, title, log_L_halo_limit=None):
    """
    Plot the abundance matched QLF (points with Poisson errors) and the best fit line QLF model (line)

    Returns the matplotlib.figure.Figure.

    :param  binned:                 binned QLF of the assigned luminosities, see binned_qlf
    :type   binned:                 dict
    :param  log_L_model:            log10 luminosities to draw the model at
    :type   log_L_model:            np.ndarray
    :param  phi_model:              model QLF at log_L_model [Mpc^-3 dex^-1]
    :type   phi_model:              np.ndarray
    :param  phi_model_in_bins:   model QLF averaged over each bin, for the ratio (model_qlf_bin_averages)
    :type   phi_model_in_bins:   np.ndarray
    :param  title:                  figure title
    :type   title:                  str
    :param  log_L_halo_limit:       faintest assigned log10 luminosity, if the sample ran out of haloes rather than
                                    reaching Log_Lmin (marked on the plot), else None
    :type   log_L_halo_limit:       float or None
    """
    fig, (ax, ax_ratio) = plt.subplots(2, 1, sharex=True, figsize=(3.21, 3.01), gridspec_kw={"height_ratios": (3, 1), "hspace": 0.02}) #

    has = binned["counts"] > 0
    x = binned["log_L_centres"][has]

    ax.plot(log_L_model, phi_model, color=model_colour, linewidth=1.5, linestyle='--',label="Shen+2020 model", zorder=-20)
    ax.errorbar(x, binned["phi"][has], yerr=binned["phi_err"][has], fmt=".", markersize=6,
                color=abundance_matched_colour, ecolor=abundance_matched_colour, elinewidth=1.,
                markeredgecolor="white", markeredgewidth=0.3, label="abundance matched BHs", zorder=-10)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\phi(L_{\rm bol})$ [Mpc$^{-3}$ dex$^{-1}$]")
    ax.set_title(title, color="k", fontsize=9, loc="left")
    legend = ax.legend(frameon=False, fontsize=8, loc="upper right", borderaxespad=0.3)


    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = binned["phi"][has] / phi_model_in_bins[has]
        ratio_err = binned["phi_err"][has] / phi_model_in_bins[has]
    ax_ratio.axhline(1., color=model_colour, linewidth=1.0, zorder=-20)
    ax_ratio.errorbar(x, ratio, yerr=ratio_err, fmt=".", markersize=6, color=abundance_matched_colour,
                      ecolor=abundance_matched_colour, elinewidth=1.0, markeredgecolor="white",
                      markeredgewidth=0.3, zorder=-10)
    
    dy=0.5
    ax_ratio.set_yticks(np.arange(0., 2+dy, dy))
    ax_ratio.set_yticks(np.arange(0., 2+dy/2, dy/2), minor=True)
    ax_ratio.set_ylim(0, 2)
    ax_ratio.set_ylabel("ratio")
    ax_ratio.set_xlabel(r"$\log_{10}\,L_{\rm bol}$ [erg s$^{-1}$]")
    
    fig.align_ylabels() # ensure subfig y-labels are aligned
    
    if log_L_halo_limit is not None:
        for a in (ax, ax_ratio):
            a.axvline(log_L_halo_limit, color=model_colour, linewidth=1.0, linestyle=":", zorder=1)

    return fig


def plot_per_shell_qlfs(shell_qlfs, log_L_model, phi_model, title):
    """
    Plot the abundance matched QLF of each lightcone shell (lines) and the model.

    Returns the matplotlib.figure.Figure.

    :param  shell_qlfs:     list of (label, binned QLF) of each shell, in increasing redshift, see binned_qlf
    :type   shell_qlfs:     list
    :param  log_L_model:    log10 luminosities to draw the model at
    :type   log_L_model:    np.ndarray
    :param  phi_model:      model QLF at log_L_model [Mpc^-3 dex^-1]
    :type   phi_model:      np.ndarray
    :param  title:          figure title
    :type   title:          str
    """
    fig, ax = plt.subplots(figsize=(3.21, 2.41))


    n_colours = 10
    #shell_colours = [shell_cmap(10) for i in range(10)] # define shell colour map, never need more than 10 shells
    # get index of shells in models and colours for plotting
    ramp_idx = np.round(np.linspace(0, n_colours-1, len(shell_qlfs))).astype(int) if len(shell_qlfs) > 1 else [4]
    for (label, binned), idx in zip(shell_qlfs, ramp_idx):
        # empty bins break the line rather than being bridged
        phi = np.where(binned["counts"] > 0, binned["phi"], np.nan)
        ax.plot(binned["log_L_centres"], phi, 
                color=shell_cmap(idx/n_colours), linewidth=1.,
                marker=".", markersize=3., label=label, zorder=-10, 
                path_effects=[path_effects.withStroke(linewidth=1.5, foreground="white"), path_effects.Normal()])
    ax.plot(log_L_model, phi_model, color=model_colour, linewidth=1.5, linestyle="--", label="Shen+2020 model", zorder=-20)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}\,L_{\rm bol}$ [erg s$^{-1}$]")
    ax.set_ylabel(r"$\phi(L_{\rm bol})$ [Mpc$^{-3}$ dex$^{-1}$]")
    ax.set_title(title, color="k", fontsize=9, loc="left")
    
    fig.align_ylabels() # ensure labels are aligned
    
    legend = ax.legend(frameon=False, fontsize=8, loc="lower left", borderaxespad=0.3,labelspacing=0.2)

    return fig


def write_table(path, binned, phi_model_in_bins, header):
    """
    Write the binned QLF and the model averaged over each bin to a text file.

    :param  path:                   output text file
    :type   path:                   str
    :param  binned:                 binned QLF, see binned_qlf
    :type   binned:                 dict
    :param  phi_model_in_bins:   model QLF averaged over each bin [Mpc^-3 dex^-1]
    :type   phi_model_in_bins:   np.ndarray
    :param  header:                 header line(s)
    :type   header:                 str
    """
    columns = np.column_stack((binned["log_L_centres"], binned["counts"], binned["phi"], binned["phi_err"], phi_model_in_bins))
    np.savetxt(path, columns, fmt=("%.3f", "%d", "%.6e", "%.6e", "%.6e"), header=header + "\nlog10_Lbol_centre[erg/s]  N  phi[Mpc^-3dex^-1]  phi_err  phi_model_bin_average")


def main(args):
    """
    Abundance match the haloes of one QLF row and plot the QLF of the assigned luminosities against the model.

    :param  args:   command line arguments, see the end of this file
    :type   args:   argparse.Namespace
    """
    # the map script's modules need the simulation readers (virgo, swiftlet, lightcone_io, healpy)
    from swiftlet.swift_snapshot_redshift_conversion import flamingo_shell_redshift_file
    from .abundance_matched_map import (get_particle_data_for_abundance_matching, sample_luminosities,
                                        assign_sampled_luminosities, report)


    
    seed = 44 # same seed as abundance_matched_map, so the same luminosities go to the same haloes
    rng = np.random.default_rng(seed)

    # define plot setting style
    plot_settings()

    particle_data, z_range, qlf_redshift, cosmo, shell_box = get_particle_data_for_abundance_matching(args)
    n_haloes = len(particle_data["Lightcone/Redshift"])

    ordered_sampled_Lbol = sample_luminosities(args, n_haloes, z_range, qlf_redshift, cosmo)
    Lbol = assign_sampled_luminosities(particle_data, ordered_sampled_Lbol, n_haloes, args.sigma_Y, rng, quiet=args.quiet)

    # the volume the luminosities were sampled for (sample_luminosities)
    volume = cosmo.lightcone_volume(z_range[0], z_range[1], use_redshift=True, phi=None).to(unyt.Mpc**3)
    n_assigned = int(np.sum(Lbol.to_value(unyt.erg / unyt.s) > 0))
    report(f"\nQLF redshift {qlf_redshift}, haloes in z = {z_range[0]:.3f} - {z_range[1]:.3f}: {n_haloes:.3e}, "
           f"with a luminosity: {n_assigned:.3e}, volume {volume:.4e}", args.quiet)

    if n_assigned == 0:
        raise ValueError(f"no luminosities were assigned to the {n_haloes} haloes of QLF row {args.qlf_shell_nr}, "
                         "there is no QLF to plot")
    log_L_assigned = np.log10(Lbol.to_value(unyt.erg / unyt.s)[Lbol.to_value(unyt.erg / unyt.s) > 0])
    log_L_edges = luminosity_bin_edges(args.Log_Lmin, log_L_assigned.min(), log_L_assigned.max() + 1e-6, args.dlogL)
    binned = binned_qlf(Lbol, volume, log_L_edges)

    # the faint end is cut by the number of haloes if every halo got a luminosity
    log_L_halo_limit = float(log_L_assigned.min()) if n_assigned == n_haloes else None

    log_L_model = np.linspace(log_L_edges[0], log_L_edges[-1], 400)
    phi_model = model_qlf(args.qlf_filename, qlf_redshift, log_L_model)
    phi_model_in_bins = model_qlf_bin_averages(args.qlf_filename, qlf_redshift, log_L_edges)

    if  args.sigma_Y < 0:
        title = (f"{args.box_res}/{args.sim_name}, QLF z = {qlf_redshift:g} ({z_range[0]:.2f}-{z_range[1]:.2f})")
    else:
        title = (f"{args.box_res}/{args.sim_name}, QLF z = {qlf_redshift:g} ({z_range[0]:.2f}-{z_range[1]:.2f})")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(args.output)[0]

    fig = plot_qlf_comparison(binned, log_L_model, phi_model, phi_model_in_bins, title, log_L_halo_limit)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_table(stem + ".txt", binned, phi_model_in_bins,
                header=f"{title}\nvolume {volume.to_value(unyt.Mpc**3):.6e} Mpc^3, {n_assigned} luminosities assigned")
    print(f"Wrote {args.output} and {stem}.txt", flush=True)

    if args.per_shell:
        shell_edges = np.loadtxt(flamingo_shell_redshift_file(shell_box), delimiter=",")
        shell_qlfs = []
        for shell_nr in np.unique(particle_data["Lightcone/ShellNumber"]):
            # part of the shell inside the QLF redshift range, the haloes outside it were not read
            z_lo = max(shell_edges[shell_nr, 0], z_range[0])
            z_hi = min(shell_edges[shell_nr, 1], z_range[1])
            if z_hi <= z_lo:
                continue
            in_shell = particle_data["Lightcone/ShellNumber"] == shell_nr
            shell_volume = cosmo.lightcone_volume(z_lo, z_hi, use_redshift=True, phi=None).to(unyt.Mpc**3)
            shell_qlfs.append((f"shell {shell_nr} (z = {z_lo:.2f} - {z_hi:.2f})",
                               binned_qlf(Lbol[in_shell], shell_volume, log_L_edges)))
        fig = plot_per_shell_qlfs(shell_qlfs, log_L_model, phi_model, title)
        fig.savefig(stem + "_shells.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {stem}_shells.png", flush=True)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Plot the QLF of the abundance matched black holes against the Shen+2020 model")

    # simulation
    argparser.add_argument("--box_res", type=str, required=True, help="boxsize and resolution, e.g. L1000N1800")
    argparser.add_argument("--sim_name", type=str, required=True, help="simulation name")
    argparser.add_argument("--lightcone_nr", type=int, default=0, help="lightcone number")

    # abundance matching, as in abundance_matched_map
    argparser.add_argument("--qlf_shell_nr", type=int, required=True, help="row of the QLF parameter file (tabulated QLF redshift) to abundance match")
    argparser.add_argument("--qlf_filename", type=str, required=True, help="path/to/qlf/data/filename")
    argparser.add_argument("--Log_Lmin", type=float, default=30, help="enforce a minimum sampled bolometric luminosity")
    argparser.add_argument("--sigma_Y", type=float, default=-1., help="scatter in dex added to the Vmax ranking, or < 0 to rank by accretion rate")

    # plot
    argparser.add_argument("--output", type=str, required=True, help="output figure (.png); the binned QLF is written next to it as .txt")
    argparser.add_argument("--dlogL", type=float, default=0.1, help="width of the log10 luminosity bins of the plotted QLF")
    argparser.add_argument("--per_shell", action="store_true", help="also plot the QLF of each lightcone shell, to <output>_shells.png")
    argparser.add_argument("--quiet", action="store_true", help="only print the files written")
    
    main(argparser.parse_args())