import streamlit as st
import numpy as np
import pandas as pd
import scanpy as sc
import io
import os
import warnings
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import colors
import plotly.figure_factory as ff
import random
import pickle
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings("ignore")

### SIDEBAR

# Chose resolution and top_n_markers (scanpy input variables)
st.sidebar.markdown("## Choose options")
resolution = st.sidebar.number_input("Choose Louvain resolution",
                value=0.5, min_value=0.0, max_value=5.0, step=0.1
                )


top_n_markers = st.sidebar.slider("Choose number of top marker genes",
                value=1000, min_value=0, max_value=10000, step=100
                )

# Round resolution (computer/float calculation mismanagement)
resolution = round(resolution, 2)
st.sidebar.write("Resolution chosen: ", resolution)
st.sidebar.write("Top number of markers chosen: ", top_n_markers)


### MAIN

# Display basic app info
st.markdown('# Streamlit app')
st.markdown("Testing the effects of different louvain resolution and number" + \
            " of marker genes used to identify different clusters in data"
)

# Display basic dataset info
st.markdown("""
## Dataset:
### Single-cell RNA-seq analysis of 1,522 cells of the small intestinal epithelium

Number of cells: 1,520

Organism: Mus musculus

Publication:
- Haber AL, Biton M, Rogel N, Herbst RH, Shekhar K et al. (2017) A single-cell survey of the small intestinal epithelium.
""")


# Show code to load data
st.markdown("## Loading pre-processed scanpy object: ")
st.code("""
adata = sc.read("./scanpy_obj.h5ad")

n_cells, n_genes = adata.shape
sc.markdown(f"n_cells x n_genes: {n_cells} x {n_genes}")
""")


# Load dataset:
#   Initiate adata object from first sample (e.g. mouse 1) and
#   then concencatenate the rest of samples to this object
input_files = os.listdir("./input")[1:]
first_input = os.listdir("./input")[0]

adata = sc.read("./input/" + first_input)
for f in input_files:
    adata_temp = sc.read("./input/" + f)
    adata = adata.concatenate(adata_temp)

with open("uns.pickle", 'rb') as handle:
    adata.uns = pickle.load(handle)

# Run dimensionality reduction functions
sc.pp.pca(adata, n_comps=50, use_highly_variable=True, svd_solver="arpack")
sc.pp.neighbors(adata, n_pcs=50)
sc.tl.umap(adata)

# Show data shape
n_cells, n_genes = adata.shape
st.markdown(f"n_cells x n_genes: {n_cells} x {n_genes}")

# Option: show SAMPLE (obs) and GENE (var) info on active checkbox
show_tables = st.checkbox("Show SAMPLE and GENE info")
if show_tables:
    st.markdown("### Observations: ")
    buffer = io.StringIO()
    adata.obs.info(buf=buffer)
    s = buffer.getvalue()

    st.text(s)

    st.markdown("### Variables: ")
    buffer = io.StringIO()
    adata.var.info(buf=buffer)
    s = buffer.getvalue()

    st.text(s)


# Show UMAP plots with computed number of clusters (based on resolution)
st.markdown("## Clustering")

# Show code used  
st.code(f"""
sc.tl.louvain(adata, resolution=1.0,
                key_added='louvain_r1',
                random_state=42)

sc.tl.louvain(adata, resolution={str(resolution)},
                key_added='louvain_r_chosen',
                random_state=42)

sc.set_figure_params(figsize=[8, 8])
st.pyplot(
    sc.pl.umap(adata,
     color = ["louvain_r1", 'louvain_r_chosen'])
)
""")

# Compute clustering values
st.text(f"Resolution: 1.0\t\t\t\t\tResolution: {resolution}")
sc.tl.louvain(adata, resolution=1.0,
                key_added='louvain_r1',
                random_state=42)

sc.tl.louvain(adata, resolution=resolution,
                key_added='louvain_r_chosen',
                random_state=42)

# Plot umap at resolution of 1 and chosen resolution
sc.set_figure_params(figsize=[8, 8])
sc.pl.umap(adata,
     color = ["louvain_r1", 'louvain_r_chosen'])
st.pyplot()


# Define dictionary of marker genes
marker_genes = {}

#marker_genes['Goblet'] = ['Agr2', 'Fcgbp', 'Tff3', 'Clca1', 'Zg16', 'Tpsg1', 'Muc2', 'Galnt12', 'Atoh1', 'Rep15', 'S100a6', 'Pdia5', 'Klk1', 'Pla2g10', 'Spdef', 'Lrrc26', 'Ccl9', 'Bace2', 'Bcas1', 'Slc12a8', 'Smim14', 'Tspan13', 'Txndc5', 'Creb3l4', 'C1galt1c1', 'Creb3l1', 'Qsox1', 'Guca2a', 'Scin', 'Ern2', 'AW112010', 'Fkbp11', 'Capn9', 'Stard3nl', 'Slc50a1', 'Sdf2l1', 'Hgfa', 'Galnt7', 'Hpd', 'Ttc39a', 'Tmed3', 'Pdia6', 'Uap1', 'Gcnt3', 'Tnfaip8', 'Dnajc10', 'Ergic1', 'Tsta3', 'Kdelr3', 'Foxa3', 'Tpd52', 'Tmed9', 'Spink4', 'Nans', 'Cmtm7', 'Creld2', 'Tm9sf3', 'Wars', 'Smim6', 'Manf', 'Oit1', 'Tram1', 'Kdelr2', 'Xbp1', 'Serp1', 'Vimp', 'Guk1', 'Sh3bgrl3', 'Cmpk1', 'Tmsb10', 'Dap', 'Ostc', 'Ssr4', 'Sec61b', 'Pdia3', 'Gale', 'Klf4', 'Krtcap2', 'Arf4', 'Sep15', 'Ssr2', 'Ramp1', 'Calr', 'Ddost']
#marker_genes['Stem'] = ['Lgr5', 'Ascl2', 'Slc12a2', 'Axin2', 'Olfm4', 'Gkn3']
#marker_genes['Paneth'] = ['Gm15284', 'AY761184', 'Defa17', 'Gm14851', 'Defa22', 'Defa-rs1', 'Defa3', 'Defa24', 'Defa26', 'Defa21', 'Lyz1', 'Gm15292', 'Mptx2', 'Ang4']
#marker_genes['Enteroendocrine'] = ['Chgb', 'Gfra3', 'Cck', 'Vwa5b2', 'Neurod1', 'Fev', 'Aplp1', 'Scgn', 'Neurog3', 'Resp18', 'Trp53i11', 'Bex2', 'Rph3al', 'Scg5', 'Pcsk1', 'Isl1', 'Maged1', 'Fabp5', 'Celf3', 'Pcsk1n', 'Fam183b', 'Prnp', 'Tac1', 'Gpx3', 'Cplx2', 'Nkx2-2', 'Olfm1', 'Vim', 'Rimbp2', 'Anxa6', 'Scg3', 'Ngfrap1', 'Insm1', 'Gng4', 'Pax6', 'Cnot6l', 'Cacna2d1', 'Tox3', 'Slc39a2', 'Riiad1']
#marker_genes['Tuft'] = ['Alox5ap', 'Lrmp', 'Hck', 'Avil', 'Rgs13', 'Ltc4s', 'Trpm5', 'Dclk1', 'Spib', 'Fyb', 'Ptpn6', 'Matk', 'Snrnp25', 'Sh2d7', 'Ly6g6f', 'Kctd12', '1810046K07Rik', 'Hpgds', 'Tuba1a', 'Pik3r5', 'Vav1', 'Tspan6', 'Skap2', 'Pygl', 'Ccdc109b', 'Ccdc28b', 'Plcg2', 'Ly6g6d', 'Alox5', 'Pou2f3', 'Gng13', 'Bmx', 'Ptpn18', 'Nebl', 'Limd2', 'Pea15a', 'Tmem176a', 'Smpx', 'Itpr2', 'Il13ra1', 'Siglecf', 'Ffar3', 'Rac2', 'Hmx2', 'Bpgm', 'Inpp5j', 'Ptgs1', 'Aldh2', 'Pik3cg', 'Cd24a', 'Ethe1', 'Inpp5d', 'Krt23', 'Gprc5c', 'Reep5', 'Csk', 'Bcl2l14', 'Tmem141', 'Coprs', 'Tmem176b', '1110007C09Rik', 'Ildr1', 'Galk1', 'Zfp428', 'Rgs2', 'Inpp5b', 'Gnai2', 'Pla2g4a', 'Acot7', 'Rbm38', 'Gga2', 'Myo1b', 'Adh1', 'Bub3', 'Sec14l1', 'Asah1', 'Ppp3ca', 'Agt', 'Gimap1', 'Krt18', 'Pim3', '2210016L21Rik', 'Tmem9', 'Lima1', 'Fam221a', 'Nt5c3', 'Atp2a3', 'Mlip', 'Vdac3', 'Ccdc23', 'Tmem45b', 'Cd47', 'Lect2', 'Pla2g16', 'Mocs2', 'Arpc5', 'Ndufaf3']
#ta_genes = "2200002D01Rik, Actb, Actg1, Adh1, Aldh1a1, Aldob, Alpi, Apoa1, Apoa4, Apoc2, Apoc3, Atp1a1, Cbr1, Cdhr2, Cdhr5, Ces2e, Chgb, Clca4, Creb3l3, Crip1, Cyb5r3, Cyp2b10, Cyp2c29, Cyp2d26, Cyp3a11, Cyp4f14, Dak, Dbi, Dgat1, Dmbt1, Eef1a1, Eef1b2, Ephx1, Fabp1, Fabp2, Fam213a, Fbp2, Fth1, Gapdh, Gm10653, Gm12070, Gm13826, Gm5766, Gm6251, Gm6402, Gm6654, Gna11, Gnb2l1, Gpd1, Gpi1, Gpx2, Gpx4, Gsta1, Gsta4, Gstm1, Gstm3, Guca2b, H2afz, Khk, Ldha, Lypd8, Malat1, Mdh2, Mgam, Mt1, Mt2, Mttp, Ndrg1, Npm1, P4hb, Papss2, Pigr, Pkm, Ppia, Prap1, Ptma, Ran, Rbp2, Reg1, Reg4, Rn45s, Rpl10, Rpl10a, Rpl11, Rpl12, Rpl13, Rpl13a, Rpl14, Rpl15, Rpl18, Rpl18a, Rpl19, Rpl23, Rpl23a, Rpl27, Rpl27a, Rpl29, Rpl3, Rpl31, Rpl31-ps12, Rpl32, Rpl35, Rpl36, Rpl36a, Rpl37, Rpl39, Rpl4, Rpl41, Rpl5, Rpl6, Rpl7, Rpl7a, Rpl8, Rplp0, Rplp1, Rps10, Rps11, Rps12, Rps13, Rps16, Rps17, Rps18, Rps19, Rps19-ps3, Rps2, Rps20, Rps21, Rps23, Rps24, Rps25, Rps26, Rps27a, Rps3a1, Rps4x, Rps5, Rps6, Rps7, Rps8, Rps9, Rpsa, S100g, Sepp1, Sis, Slc12a2, Slc5a1, Snrpg, Spink3, Tmsb4x, Tuba1b, Ucp2"
#marker_genes["TA"] = ta_genes.split(", ")


# Load marker genes from Haber et al. 2018
marker_df = pd.read_csv("./marker_genes.csv")

# Reorder columns (better first cell types for visual display)
marker_df = marker_df[marker_df.columns.tolist()[::-1]]
for col in marker_df.columns:
    arr = marker_df[col].astype(str).values
    arr = arr[arr != "nan"].tolist()
    
    if col in marker_genes.keys():
        marker_genes[col] += arr
        
    else:
        if len(arr) > 0:
            if col == "TA (G2)":
                marker_genes["TA"] = arr
            else:
                marker_genes[col] = arr

            
# Load marker genes from CellMarker database (Zhang et al. 2018)
marker_df = pd.read_csv("./marker_df.csv", index_col=0)

# Reorder columns (better first cell types for visual display)
marker_df = marker_df[marker_df.columns.tolist()[::-1]]
for col in marker_df.columns:
    arr = marker_df[col].astype(str).values
    arr = arr[arr != "nan"].tolist()
    
    if col in marker_genes.keys():
        marker_genes[col] += arr
        
    else:
        if len(arr) > 0:
            if col == "TA (G2)":
                marker_genes["TA"] = arr
            else:
                marker_genes[col] = arr


# To annotate clusters, we need to check marker gene expression for each cell type
st.markdown("## Cluster annotation")
st.markdown("Testing proportion of marker genes for different intestinal cell types that were expressed in each cluster at chosen resolution")

# Show code to obtain input
st.code(f"""
#Calculate marker genes
sc.tl.rank_genes_groups(adata, groupby='louvain_r_chosen',
                         key_added='rank_genes_r_chosen')
cell_annotation = sc.tl.marker_gene_overlap(adata, marker_genes,
                                            key='rank_genes_r_chosen')


cell_annotation_norm = sc.tl.marker_gene_overlap(adata,
                                                 marker_genes, key='rank_genes_r_chosen',
                                                 normalize='reference',
                                                 top_n_markers = {str(top_n_markers)})

cell_annotation_norm = cell_annotation_norm.loc[cell_annotation_norm.index.tolist()[::-1]] 
vals = cell_annotation_norm.values.round(2)
fig = ff.create_annotated_heatmap(y=cell_annotation_norm.index.tolist(),
        x=["Cluster " + str(x) + "<br>" for x in cell_annotation_norm.columns.tolist()],
        reversescale = True, showscale=True,
        z=vals, colorscale="blues")

fig.update_layout(width=800, height=400)
st.plotly_chart(fig)
""")

#Calculate marker genes
sc.tl.rank_genes_groups(adata, groupby='louvain_r_chosen',
                         key_added='rank_genes_r_chosen')
cell_annotation = sc.tl.marker_gene_overlap(adata, marker_genes, 
                                            key='rank_genes_r_chosen')


cell_annotation_norm = sc.tl.marker_gene_overlap(adata,
                                                 marker_genes, key='rank_genes_r_chosen',
                                                 normalize='reference',
                                                 top_n_markers = top_n_markers)



cell_annotation_norm = cell_annotation_norm.loc[cell_annotation_norm.index.tolist()[::-1]] 
vals = cell_annotation_norm.values.round(2)

# Plot heatmap of ratios of marker genes that were upregulated in each cluster
fig = ff.create_annotated_heatmap(y=cell_annotation_norm.index.tolist(),
                                  x=["Cluster " + str(x) + "<br>" for x in cell_annotation_norm.columns.tolist()],
                                  reversescale = True, showscale=True,
                                  z=vals, colorscale="blues")

fig.update_layout(width=800, height=400)
st.plotly_chart(fig)

# Check specific marker gene expression, compare at each cell type level
st.markdown("## Check marker gene expression")
st.markdown("- The marker gene data used was obtained by combining the data from [Haber et al. 2018](https://www.nature.com/articles/nature24489 'Go to paper') " + \
    "and the CellMarker database ([Zhang et al. 2018](https://doi.org/10.1093/nar/gky900 'Go to paper'))")

#Define a nice colour map for gene expression
colors2 = plt.cm.Reds(np.linspace(0, 1, 128))
colors3 = plt.cm.Greys_r(np.linspace(0.7,0.8,20))
colorsComb = np.vstack([colors3, colors2])
mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colorsComb)

# Based on user input, show downstream data for a specific cell type
# for which marker genes are available
celltype = st.selectbox(options = list(marker_genes.keys()), label="Choose cell type")
celltype_colname = f'{celltype}_marker_expr'
paneth_genes = adata.var_names[np.in1d(adata.var_names, marker_genes[celltype])]
adata.obs[celltype_colname] = adata[:,paneth_genes].X.mean(1)

st.markdown("- Averaged marker expression in each cluster")
#plt.rcParams["figure.figsize"] = (8, 8)
# Create two columns for plots and compare marker gene expression with a violin plot
#   to umap clusters to see where the cell type could be annotated
col1, col2 = st.beta_columns(2)
sc.pl.violin(adata, celltype_colname, groupby='louvain_r_chosen')
col1.pyplot()
sc.pl.umap(adata, color = ['louvain_r_chosen'])
col2.pyplot()

# Show individual marker genes (n=9) expression on the UMAP plot to see how well
#  it matches expected cluster position
st.markdown("- Individual random marker gene expression from chosen celltype")
reload = st.button("Show expression of random marker genes/Reload different genes")

# Function to load/reload genes
def load_genes():
    #Check individual markers
    chosen_marker_genes = adata.var_names[np.in1d(adata.var_names, marker_genes[celltype])]
    plotted = 0
    while plotted < 3:
        genes = random.sample(chosen_marker_genes.tolist(), 3)

        try:
            sc.pl.umap(adata, color=genes, title=genes, color_map=mymap)
            st.pyplot()
        except IndexError:
            continue

        plotted += 1

# When the button is clicked: load/reload the marker gene plots
if reload == True:
    load_genes()

