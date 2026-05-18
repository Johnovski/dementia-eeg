"""
Dataset Demographics Analysis & Visualization
DS004504 - EEG Dementia Classification Dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ──
df = pd.read_csv('/Users/john/Dementia/ds004504/participants.tsv', sep='\t')
df.columns = df.columns.str.strip()
df['Group'] = df['Group'].str.strip()
group_map = {'A': 'AD', 'F': 'FTD', 'C': 'CN'}
df['Diagnosis'] = df['Group'].map(group_map)
df['Gender'] = df['Gender'].str.strip()

# Colors
colors = {'AD': '#E74C3C', 'FTD': '#F39C12', 'CN': '#27AE60'}
gender_colors = {'M': '#3498DB', 'F': '#E91E8A'}

print("="*60)
print("DATASET SUMMARY — DS004504")
print("="*60)
print(f"Total subjects: {len(df)}")
for g in ['AD', 'FTD', 'CN']:
    sub = df[df['Diagnosis'] == g]
    print(f"\n{g} (n={len(sub)}):")
    print(f"  Age:  {sub['Age'].mean():.1f} ± {sub['Age'].std():.1f} (range: {sub['Age'].min()}-{sub['Age'].max()})")
    print(f"  MMSE: {sub['MMSE'].mean():.1f} ± {sub['MMSE'].std():.1f} (range: {sub['MMSE'].min()}-{sub['MMSE'].max()})")
    print(f"  Gender: M={len(sub[sub['Gender']=='M'])}, F={len(sub[sub['Gender']=='F'])}")

# ══════════════════════════════════════════════════════════════
# FIGURE 1: Comprehensive Demographics Dashboard
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#FAFAFA')
fig.suptitle('Dataset Demographics — DS004504\nEEG Resting-State Dementia Classification',
             fontsize=18, fontweight='bold', y=0.98, color='#2C3E50')

gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3, top=0.91, bottom=0.08, left=0.07, right=0.95)

# ── 1. Group Distribution (Donut Chart) ──
ax1 = fig.add_subplot(gs[0, 0])
counts = [len(df[df['Diagnosis']==g]) for g in ['AD', 'FTD', 'CN']]
labels = ['AD (n=36)', 'FTD (n=23)', 'CN (n=29)']
wedges, texts, autotexts = ax1.pie(counts, labels=labels,
    colors=[colors['AD'], colors['FTD'], colors['CN']],
    autopct='%1.1f%%', startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
for t in autotexts:
    t.set_fontsize(11)
    t.set_fontweight('bold')
    t.set_color('white')
for t in texts:
    t.set_fontsize(10)
ax1.set_title('Group Distribution', fontsize=13, fontweight='bold', pad=15)

# ── 2. Age Distribution by Group ──
ax2 = fig.add_subplot(gs[0, 1])
positions = [1, 2, 3]
for i, g in enumerate(['AD', 'FTD', 'CN']):
    data = df[df['Diagnosis']==g]['Age']
    bp = ax2.boxplot([data], positions=[positions[i]], widths=0.5,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='white', markersize=6),
                     medianprops=dict(color='white', linewidth=2),
                     flierprops=dict(marker='o', markerfacecolor=colors[g], alpha=0.5))
    bp['boxes'][0].set_facecolor(colors[g])
    bp['boxes'][0].set_alpha(0.8)
    # Scatter individual points
    jitter = np.random.normal(0, 0.06, len(data))
    ax2.scatter(positions[i] + jitter, data, color=colors[g], alpha=0.4, s=25, zorder=5)

ax2.set_xticks(positions)
ax2.set_xticklabels(['AD', 'FTD', 'CN'], fontsize=11)
ax2.set_ylabel('Age (years)', fontsize=11)
ax2.set_title('Age Distribution by Group', fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ── 3. MMSE Distribution by Group ──
ax3 = fig.add_subplot(gs[0, 2])
for i, g in enumerate(['AD', 'FTD', 'CN']):
    data = df[df['Diagnosis']==g]['MMSE']
    bp = ax3.boxplot([data], positions=[positions[i]], widths=0.5,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='white', markersize=6),
                     medianprops=dict(color='white', linewidth=2),
                     flierprops=dict(marker='o', markerfacecolor=colors[g], alpha=0.5))
    bp['boxes'][0].set_facecolor(colors[g])
    bp['boxes'][0].set_alpha(0.8)
    jitter = np.random.normal(0, 0.06, len(data))
    ax3.scatter(positions[i] + jitter, data, color=colors[g], alpha=0.4, s=25, zorder=5)

ax3.axhline(y=24, color='gray', linestyle='--', alpha=0.6, label='MMSE ≤ 24 = impairment')
ax3.legend(fontsize=9, loc='lower left')
ax3.set_xticks(positions)
ax3.set_xticklabels(['AD', 'FTD', 'CN'], fontsize=11)
ax3.set_ylabel('MMSE Score', fontsize=11)
ax3.set_title('MMSE Distribution by Group', fontsize=13, fontweight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ── 4. Gender Distribution per Group ──
ax4 = fig.add_subplot(gs[1, 0])
groups_order = ['AD', 'FTD', 'CN']
male_counts = [len(df[(df['Diagnosis']==g) & (df['Gender']=='M')]) for g in groups_order]
female_counts = [len(df[(df['Diagnosis']==g) & (df['Gender']=='F')]) for g in groups_order]
x = np.arange(len(groups_order))
w = 0.35
bars1 = ax4.bar(x - w/2, male_counts, w, label='Male', color='#3498DB', alpha=0.85, edgecolor='white')
bars2 = ax4.bar(x + w/2, female_counts, w, label='Female', color='#E91E8A', alpha=0.85, edgecolor='white')
for bar in bars1:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, int(bar.get_height()),
             ha='center', fontsize=10, fontweight='bold', color='#3498DB')
for bar in bars2:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, int(bar.get_height()),
             ha='center', fontsize=10, fontweight='bold', color='#E91E8A')
ax4.set_xticks(x)
ax4.set_xticklabels(groups_order, fontsize=11)
ax4.set_ylabel('Count', fontsize=11)
ax4.set_title('Gender Distribution per Group', fontsize=13, fontweight='bold', pad=15)
ax4.legend(fontsize=10)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.grid(axis='y', alpha=0.3)

# ── 5. Age vs MMSE Scatter ──
ax5 = fig.add_subplot(gs[1, 1])
for g in ['AD', 'FTD', 'CN']:
    sub = df[df['Diagnosis']==g]
    ax5.scatter(sub['Age'], sub['MMSE'], c=colors[g], label=g, s=60, alpha=0.7,
                edgecolors='white', linewidth=0.5)
ax5.axhline(y=24, color='gray', linestyle='--', alpha=0.5)
ax5.fill_between([40, 85], 0, 24, color='#ffcccc', alpha=0.15, label='Cognitive impairment zone')
ax5.set_xlabel('Age (years)', fontsize=11)
ax5.set_ylabel('MMSE Score', fontsize=11)
ax5.set_title('Age vs MMSE by Diagnosis', fontsize=13, fontweight='bold', pad=15)
ax5.legend(fontsize=9, loc='lower left')
ax5.set_xlim(40, 85)
ax5.set_ylim(0, 33)
ax5.grid(alpha=0.3)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# ── 6. Summary Statistics Table ──
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
table_data = []
for g in ['AD', 'FTD', 'CN']:
    sub = df[df['Diagnosis']==g]
    m = len(sub[sub['Gender']=='M'])
    f = len(sub[sub['Gender']=='F'])
    table_data.append([
        g, str(len(sub)),
        f"{sub['Age'].mean():.1f} ± {sub['Age'].std():.1f}",
        f"{m}/{f}",
        f"{sub['MMSE'].mean():.1f} ± {sub['MMSE'].std():.1f}",
        f"{sub['MMSE'].min()}–{sub['MMSE'].max()}"
    ])
col_labels = ['Group', 'N', 'Age (μ±σ)', 'M/F', 'MMSE (μ±σ)', 'MMSE Range']
table = ax6.table(cellText=table_data, colLabels=col_labels, loc='center',
                  cellLoc='center', colColours=['#ECF0F1']*6)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 1.8)
# Color the group column cells
for i, g in enumerate(['AD', 'FTD', 'CN']):
    table[i+1, 0].set_facecolor(colors[g])
    table[i+1, 0].set_text_props(color='white', fontweight='bold')
ax6.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=25)

plt.savefig('/Users/john/Dementia/figures/dataset_demographics.png', dpi=200, bbox_inches='tight',
            facecolor='#FAFAFA')
print("\n✅ Saved: figures/dataset_demographics.png")

# ══════════════════════════════════════════════════════════════
# FIGURE 2: MMSE Severity Histogram
# ══════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(14, 6))
fig2.patch.set_facecolor('#FAFAFA')
bins = np.arange(0, 33, 2)
for g in ['AD', 'FTD', 'CN']:
    data = df[df['Diagnosis']==g]['MMSE']
    ax.hist(data, bins=bins, alpha=0.6, color=colors[g], label=f'{g} (n={len(data)})',
            edgecolor='white', linewidth=0.8)

ax.axvline(x=24, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.7)
ax.text(24.3, ax.get_ylim()[1]*0.9, 'MMSE ≤ 24\n= Cognitive\nImpairment',
        fontsize=9, color='#E74C3C', fontweight='bold')
ax.set_xlabel('MMSE Score', fontsize=12)
ax.set_ylabel('Number of Subjects', fontsize=12)
ax.set_title('MMSE Score Distribution Across Groups', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/john/Dementia/figures/dataset_mmse_histogram.png', dpi=200, bbox_inches='tight',
            facecolor='#FAFAFA')
print("✅ Saved: figures/dataset_mmse_histogram.png")

# ══════════════════════════════════════════════════════════════
# FIGURE 3: Individual Subject Profile (sorted by MMSE)
# ══════════════════════════════════════════════════════════════
fig3, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(20, 8), sharex=True,
                                       gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.08})
fig3.patch.set_facecolor('#FAFAFA')

df_sorted = df.sort_values(['Diagnosis', 'MMSE'], key=lambda x: x.map({'AD': 0, 'FTD': 1, 'CN': 2}) if x.name == 'Diagnosis' else x)
df_sorted = df_sorted.reset_index(drop=True)

bar_colors = [colors[d] for d in df_sorted['Diagnosis']]
x_pos = np.arange(len(df_sorted))

# Top: Age
ax_top.bar(x_pos, df_sorted['Age'], color=bar_colors, alpha=0.75, edgecolor='white', linewidth=0.3)
ax_top.set_ylabel('Age (years)', fontsize=11)
ax_top.set_title('Per-Subject Demographics: Age & MMSE (sorted by group, then MMSE)',
                 fontsize=14, fontweight='bold')
ax_top.axhline(y=df['Age'].mean(), color='gray', linestyle=':', alpha=0.5)
ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)
ax_top.grid(axis='y', alpha=0.2)

# Bottom: MMSE
ax_bot.bar(x_pos, df_sorted['MMSE'], color=bar_colors, alpha=0.75, edgecolor='white', linewidth=0.3)
ax_bot.axhline(y=24, color='gray', linestyle='--', alpha=0.6)
ax_bot.set_ylabel('MMSE Score', fontsize=11)
ax_bot.set_xlabel('Subject (sorted)', fontsize=11)
ax_bot.spines['top'].set_visible(False)
ax_bot.spines['right'].set_visible(False)
ax_bot.grid(axis='y', alpha=0.2)

# Group separators
ad_end = len(df_sorted[df_sorted['Diagnosis']=='AD'])
ftd_end = ad_end + len(df_sorted[df_sorted['Diagnosis']=='FTD'])
for ax in [ax_top, ax_bot]:
    ax.axvline(x=ad_end - 0.5, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=ftd_end - 0.5, color='black', linestyle='--', alpha=0.3)
ax_top.text(ad_end/2, ax_top.get_ylim()[1]*0.95, 'AD (36)', ha='center', fontsize=11,
            color=colors['AD'], fontweight='bold')
ax_top.text(ad_end + (ftd_end-ad_end)/2, ax_top.get_ylim()[1]*0.95, 'FTD (23)', ha='center',
            fontsize=11, color=colors['FTD'], fontweight='bold')
ax_top.text(ftd_end + (len(df_sorted)-ftd_end)/2, ax_top.get_ylim()[1]*0.95, 'CN (29)', ha='center',
            fontsize=11, color=colors['CN'], fontweight='bold')

ax_bot.set_xticks(x_pos[::5])
ax_bot.set_xticklabels(df_sorted['participant_id'].iloc[::5], rotation=45, fontsize=7)

plt.savefig('/Users/john/Dementia/figures/dataset_subject_profiles.png', dpi=200, bbox_inches='tight',
            facecolor='#FAFAFA')
print("✅ Saved: figures/dataset_subject_profiles.png")

# ══════════════════════════════════════════════════════════════
# FIGURE 4: Age distribution (violin + strip)
# ══════════════════════════════════════════════════════════════
fig4, axes = plt.subplots(1, 2, figsize=(16, 6))
fig4.patch.set_facecolor('#FAFAFA')

# Left: Age violin
for i, g in enumerate(['AD', 'FTD', 'CN']):
    data = df[df['Diagnosis']==g]['Age'].values
    parts = axes[0].violinplot([data], positions=[i], showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor(colors[g])
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('white')
    for key in ['cbars', 'cmins', 'cmaxes']:
        parts[key].set_color(colors[g])
    jitter = np.random.normal(0, 0.04, len(data))
    axes[0].scatter(i + jitter, data, color=colors[g], alpha=0.5, s=30, zorder=5,
                    edgecolors='white', linewidth=0.3)

axes[0].set_xticks([0, 1, 2])
axes[0].set_xticklabels(['AD', 'FTD', 'CN'], fontsize=12)
axes[0].set_ylabel('Age (years)', fontsize=12)
axes[0].set_title('Age Distribution (Violin + Strip)', fontsize=13, fontweight='bold')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].grid(axis='y', alpha=0.3)

# Right: Gender pie per group
for i, g in enumerate(['AD', 'FTD', 'CN']):
    sub = df[df['Diagnosis']==g]
    m = len(sub[sub['Gender']=='M'])
    f = len(sub[sub['Gender']=='F'])
    inset = axes[1].inset_axes([i*0.33 + 0.02, 0.1, 0.28, 0.7])
    wedges, _, autotexts = inset.pie([m, f], labels=['M', 'F'],
        colors=[gender_colors['M'], gender_colors['F']],
        autopct='%1.0f%%', startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=1.5))
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    inset.set_title(f'{g}\n(n={len(sub)})', fontsize=11, fontweight='bold', color=colors[g])
axes[1].axis('off')
axes[1].set_title('Gender Ratio per Group', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/john/Dementia/figures/dataset_age_gender.png', dpi=200, bbox_inches='tight',
            facecolor='#FAFAFA')
print("✅ Saved: figures/dataset_age_gender.png")

plt.close('all')
print("\n✅ All 4 dataset visualizations generated successfully!")
