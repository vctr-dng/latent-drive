# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "marimo>=0.20.4,<1",
#     "matplotlib==3.11.2",
#     "pandas==3.0.6",
#     "pillow==12.3.0",
#     "pyarrow==22.0.0",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import json
    import subprocess
    import tomllib
    from io import BytesIO
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image

    return BytesIO, Image, Path, json, mo, pd, plt, subprocess, tomllib


@app.cell(hide_code=True)
def _(Path, tomllib):
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "representation_audit.toml"
    with config_path.open("rb") as config_file:
        audit_config = tomllib.load(config_file)
    model_slug = audit_config["encoder"]["model_id"].replace("/", "__")
    artifact_root = project_root / audit_config["paths"]["artifacts_dir"]
    analysis_dir = artifact_root / "analysis" / model_slug / audit_config["analysis"]["name"]
    feature_dir = artifact_root / "features" / model_slug
    return analysis_dir, audit_config, feature_dir


@app.cell(hide_code=True)
def _(analysis_dir, feature_dir, json, pd):
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    pca_coordinates = pd.read_parquet(analysis_dir / "pca_coordinates.parquet")
    pca_variance = pd.read_parquet(analysis_dir / "pca_variance.parquet")
    pair_similarities = pd.read_parquet(analysis_dir / "pair_similarities.parquet")
    centroid_distances = pd.read_parquet(analysis_dir / "centroid_distances.parquet")
    retrievals = pd.read_parquet(analysis_dir / "retrievals.parquet")
    outliers = pd.read_parquet(analysis_dir / "outliers.parquet")
    feature_metadata = pd.read_parquet(feature_dir / "metadata.parquet")
    return (
        centroid_distances,
        feature_metadata,
        outliers,
        pair_similarities,
        pca_coordinates,
        pca_variance,
        retrievals,
        summary,
    )


@app.cell(hide_code=True)
def _(mo, summary):
    mo.md(f"""
    # V-JEPA Representation Audit

    A reactive audit of **{summary["clip_count"]:,} passive dashcam clips** encoded by
    `{summary["model_id"]}`. This notebook examines latent geometry, temporal concentration,
    session drift, nearest-neighbor retrieval, and local-density outliers.

    > These diagnostics describe this corpus. They do not establish semantic understanding,
    > causal structure, driving intent, or control performance.
    """)
    return


@app.cell(hide_code=True)
def _(mo, outliers, pca_coordinates, retrievals):
    split_options = ["all", *sorted(pca_coordinates["split"].astype(str).unique())]
    color_options = ["split", "video_id", "session_id"]
    centroid_options = ["split", "session_id", "video_id"]
    similarity_options = [
        "adjacent",
        "same_video_nonlocal",
        "same_session_cross_video",
        "cross_session",
    ]
    query_options = retrievals["query_clip_id"].drop_duplicates().astype(str).tolist()

    split_filter = mo.ui.dropdown(split_options, value="all", label="PCA split")
    color_by = mo.ui.dropdown(color_options, value="split", label="Color PCA by")
    centroid_group = mo.ui.dropdown(centroid_options, value="session_id", label="Centroids")
    similarity_filter = mo.ui.multiselect(
        similarity_options,
        value=similarity_options,
        label="Similarity baselines",
    )
    outlier_count = mo.ui.slider(
        start=5,
        stop=len(outliers),
        step=5,
        value=min(10, len(outliers)),
        label="Outliers shown",
    )
    query_clip = mo.ui.dropdown(query_options, value=query_options[0], label="Retrieval query")

    mo.vstack(
        [
            mo.md("## Controls"),
            mo.hstack(
                [split_filter, color_by, centroid_group, outlier_count],
                widths="equal",
                gap=1,
            ),
            similarity_filter,
            query_clip,
        ]
    )
    return (
        centroid_group,
        color_by,
        outlier_count,
        query_clip,
        similarity_filter,
        split_filter,
    )


@app.cell(hide_code=True)
def _(mo, summary):
    nearest_summary = summary["nearest_neighbor"]
    mo.md(
        f"""
        <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0.8rem">
          <div style="padding:1rem;border:1px solid #d8d5cd;background:#fff">
            <small>CLIPS</small><br><strong style="font-size:1.8rem">
              {summary["clip_count"]:,}
            </strong>
          </div>
          <div style="padding:1rem;border:1px solid #d8d5cd;background:#fff">
            <small>DIMENSIONS</small><br><strong style="font-size:1.8rem">
              {summary["embedding_dimension"]:,}
            </strong>
          </div>
          <div style="padding:1rem;border:1px solid #d8d5cd;background:#fff">
            <small>SAME-VIDEO TOP-1</small><br><strong style="font-size:1.8rem">
              {nearest_summary["same_video_fraction"]:.1%}
            </strong>
          </div>
          <div style="padding:1rem;border:1px solid #d8d5cd;background:#fff">
            <small>WITHIN 10 SECONDS</small><br><strong style="font-size:1.8rem">
              {nearest_summary["local_temporal_fraction"]:.1%}
            </strong>
          </div>
        </div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Latent Geometry
    """)
    return


@app.cell
def _(pca_coordinates, split_filter):
    pca_view = (
        pca_coordinates
        if split_filter.value == "all"
        else pca_coordinates[pca_coordinates["split"] == split_filter.value]
    )
    return (pca_view,)


@app.cell(hide_code=True)
def _(color_by, pca_view, plt):
    pca_figure, pca_axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    for pca_group_name, pca_group in pca_view.groupby(color_by.value, sort=True):
        pca_axis.scatter(
            pca_group["pc1"],
            pca_group["pc2"],
            s=16,
            alpha=0.58,
            label=str(pca_group_name),
            linewidths=0,
        )
    pca_axis.set(
        title=f"PCA projection colored by {color_by.value.replace('_', ' ')}",
        xlabel="Principal component 1",
        ylabel="Principal component 2",
    )
    pca_axis.spines[["top", "right"]].set_visible(False)
    pca_axis.grid(alpha=0.25)
    pca_axis.legend(frameon=False, fontsize=8, markerscale=1.5)
    pca_figure  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo, pca_variance, plt, summary):
    variance_figure, variance_axis = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    variance_axis.bar(
        pca_variance["component"],
        pca_variance["explained_variance_ratio"],
        color="#9cc3cf",
        alpha=0.7,
        label="Per component",
    )
    variance_axis.plot(
        pca_variance["component"],
        pca_variance["cumulative_explained_variance_ratio"],
        color="#176b87",
        linewidth=2.5,
        label="Cumulative",
    )
    variance_axis.axhline(0.9, color="#983f5a", linestyle="--", linewidth=1.2)
    variance_axis.set(
        title="PCA variance profile",
        xlabel="Component",
        ylabel="Explained variance ratio",
        ylim=(0, 1.02),
    )
    variance_axis.spines[["top", "right"]].set_visible(False)
    variance_axis.grid(alpha=0.25)
    variance_axis.legend(frameon=False)
    mo.vstack(
        [
            variance_figure,
            mo.md(
                f"The first two components explain "
                f"**{summary['pca']['first_two_explained_variance']:.1%}** of variance; "
                f"**{summary['pca']['components_for_90_percent']} components** reach 90%."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Similarity Structure
    """)
    return


@app.cell(hide_code=True)
def _(pair_similarities, similarity_filter):
    similarity_view = pair_similarities[pair_similarities["category"].isin(similarity_filter.value)]
    return (similarity_view,)


@app.cell(hide_code=True)
def _(plt, similarity_view):
    category_labels = {
        "adjacent": "Adjacent clips",
        "same_video_nonlocal": "Same video, nonlocal",
        "same_session_cross_video": "Same session, other video",
        "cross_session": "Cross-session",
    }
    category_order = [name for name in category_labels if name in set(similarity_view["category"])]
    similarity_values = [
        similarity_view.loc[
            similarity_view["category"] == category_name,
            "cosine_similarity",
        ].to_numpy()
        for category_name in category_order
    ]
    similarity_figure, similarity_axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    similarity_parts = similarity_axis.violinplot(
        similarity_values,
        showmeans=False,
        showmedians=True,
        widths=0.8,
    )
    for similarity_body in similarity_parts["bodies"]:
        similarity_body.set_facecolor("#176b87")
        similarity_body.set_edgecolor("#0e3f50")
        similarity_body.set_alpha(0.65)
    similarity_parts["cmedians"].set_color("#983f5a")
    similarity_axis.set_xticks(
        range(1, len(category_order) + 1),
        [category_labels[name] for name in category_order],
        rotation=10,
    )
    similarity_axis.set(
        title="Pairwise similarity by relationship",
        ylabel="Cosine similarity",
    )
    similarity_axis.spines[["top", "right"]].set_visible(False)
    similarity_axis.grid(alpha=0.25)
    similarity_figure  # noqa: B018
    return


@app.cell(hide_code=True)
def _(summary):
    neighbor_metrics = summary["nearest_neighbor"]
    neighbor_labels = ["Same video", "Within 10 s", "Same session", "Same split"]
    neighbor_values = [
        neighbor_metrics["same_video_fraction"],
        neighbor_metrics["local_temporal_fraction"],
        neighbor_metrics["same_session_fraction"],
        neighbor_metrics["same_split_fraction"],
    ]
    return neighbor_labels, neighbor_values


@app.cell(hide_code=True)
def _(neighbor_labels, neighbor_values, plt):
    neighbor_figure, neighbor_axis = plt.subplots(figsize=(9, 4.4), constrained_layout=True)
    neighbor_bars = neighbor_axis.barh(
        neighbor_labels,
        neighbor_values,
        color=["#176b87", "#4f91a7", "#d08c28", "#983f5a"],
    )
    neighbor_axis.bar_label(
        neighbor_bars,
        labels=[f"{value:.1%}" for value in neighbor_values],
        padding=5,
    )
    neighbor_axis.set(
        title="Top-1 neighbor concentration",
        xlabel="Fraction of clips",
        xlim=(0, 1.08),
    )
    neighbor_axis.spines[["top", "right"]].set_visible(False)
    neighbor_axis.grid(alpha=0.25)
    neighbor_figure  # noqa: B018
    return


@app.cell(hide_code=True)
def _(centroid_distances, centroid_group, mo):
    centroid_view = centroid_distances[centroid_distances["group_type"] == centroid_group.value][
        ["left", "right", "cosine_similarity", "cosine_distance"]
    ].sort_values("cosine_distance", ascending=False)
    mo.vstack(
        [
            mo.md("### Centroid distances"),
            mo.ui.table(centroid_view, pagination=True, page_size=12),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Local-Density Outliers

    These clips have the largest mean cosine distance to their nearest neighbors. They are
    candidates for inspection, not automatically errors or rare semantic events.
    """)
    return


@app.cell(hide_code=True)
def _(mo, outlier_count, outliers):
    outlier_view = outliers.head(outlier_count.value)[
        ["clip_id", "video_id", "split", "start_seconds", "mean_neighbor_cosine_distance"]
    ]
    mo.ui.table(outlier_view, pagination=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Representative Retrieval

    Change the query in **Controls** to inspect another deterministic audit query. Frames are
    decoded at each clip midpoint; cosine similarity is computed from normalized embeddings.
    """)
    return


@app.cell(hide_code=True)
def _(
    BytesIO,
    Image,
    audit_config,
    feature_metadata,
    mo,
    query_clip,
    retrievals,
    subprocess,
):
    selected_retrievals = retrievals[retrievals["query_clip_id"] == query_clip.value].sort_values(
        "rank"
    )
    selected_clip_ids = [
        query_clip.value,
        *selected_retrievals["neighbor_clip_id"].astype(str).tolist(),
    ]
    metadata_lookup = feature_metadata.set_index("clip_id")
    clip_duration = float(audit_config["clips"]["duration_seconds"])

    def decode_midpoint(clip_id: str):
        clip = metadata_lookup.loc[clip_id]
        timestamp = float(clip["start_seconds"]) + clip_duration / 2.0
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(timestamp),
            "-i",
            str(clip["path"]),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ]
        result = subprocess.run(command, check=True, capture_output=True)
        return Image.open(BytesIO(result.stdout)).copy()

    selected_images = [decode_midpoint(clip_id) for clip_id in selected_clip_ids]
    retrieval_cards = []
    for card_index, (clip_id, image) in enumerate(
        zip(selected_clip_ids, selected_images, strict=True)
    ):
        if card_index == 0:
            card_title = "QUERY"
            card_detail = clip_id
        else:
            retrieval_row = selected_retrievals.iloc[card_index - 1]
            card_title = f"NEIGHBOR {card_index}"
            card_detail = f"{clip_id} | cosine {retrieval_row['cosine_similarity']:.3f}"
        retrieval_cards.append(
            mo.vstack(
                [
                    mo.md(f"**{card_title}**"),
                    mo.image(image, width=260),
                    mo.md(f"`{card_detail}`"),
                ]
            )
        )
    mo.vstack(
        [
            mo.hstack(retrieval_cards, wrap=True, gap=1),
            mo.ui.table(
                selected_retrievals[
                    [
                        "rank",
                        "neighbor_clip_id",
                        "neighbor_split",
                        "cosine_similarity",
                        "same_video",
                        "same_session",
                    ]
                ],
                pagination=False,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, summary):
    adjacent_median = summary["pair_similarity"]["adjacent"]["median"]
    nonlocal_median = summary["pair_similarity"]["same_video_nonlocal"]["median"]
    mo.md(
        f"""
        ## Reading the Audit

        - The first two principal components explain
          **{summary["pca"]["first_two_explained_variance"]:.1%}** of measured variance, so the
          projection is a map rather than the complete representation.
        - **{summary["nearest_neighbor"]["local_temporal_fraction"]:.1%}** of top-1 neighbors are
          from the same video within 10 seconds, showing strong short-range concentration.
        - Median adjacent-clip similarity is **{adjacent_median:.3f}**, versus
          **{nonlocal_median:.3f}** for nonlocal clips from the same video.
        - PCA and retrieval use the full corpus for descriptive inspection only. A downstream
          evaluation requires task labels and an evaluation protocol defined independently of
          these visualizations.

        Analysis revision: `{summary["model_revision"]}`
        Reproducibility seed: `{summary["analysis_config"]["seed"]}`
        """
    )
    return


if __name__ == "__main__":
    app.run()
