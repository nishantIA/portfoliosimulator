#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# App Title
st.title('Venture Capital Fund Simulator')

# Sidebar inputs
stages = ['Pre-Seed', 'Seed', 'Series A', 'Series B']
st.sidebar.header('Fund Parameters')
fund_size = st.sidebar.slider('Fund Size ($MM)', 10, 500, 100, step=5)
initial_stage = st.sidebar.selectbox('Initial Investment Stage', stages)
stage_index = stages.index(initial_stage)

# Robust Portfolio Allocation
st.sidebar.header('Portfolio Allocation (%) per Stage')
valid_stages = stages[stage_index:]
stage_allocations = {}
allocation_values = []
remaining_alloc = 100

for i, stage in enumerate(valid_stages):
    if i == len(valid_stages) - 1:
        allocation = remaining_alloc
        st.sidebar.write(f"Allocation to {stage}: {allocation}% (auto-set)")
    else:
        max_alloc = remaining_alloc
        if max_alloc == 0:
            allocation = 0
            st.sidebar.write(f"Allocation to {stage}: 0% (auto-set since fully allocated)")
        else:
            allocation = st.sidebar.slider(
                f'Allocation to {stage} (%)',
                min_value=0,
                max_value=max_alloc,
                value=min(25, max_alloc),
                step=5
            )
        remaining_alloc -= allocation
    allocation_values.append(allocation)
    stage_allocations[stage] = allocation

if sum(allocation_values) != 100:
    st.sidebar.warning(f"Total allocation is {sum(allocation_values)}%. Adjust allocations to total exactly 100%.")

st.sidebar.header('Entry Valuations and Check Sizes per Stage ($MM)')
valuations, check_sizes = {}, {}
for stage in valid_stages:
    valuations[stage] = st.sidebar.slider(f'Entry Valuation Range {stage}', 1, 200, (5, 20), step=1)
    check_sizes[stage] = st.sidebar.slider(f'Check Size Range {stage}', 0.25, 15.0, (1.0, 5.0), step=0.25)

st.sidebar.header('Stage Progression Probabilities (%)')
prob_advancement = {}
for i in range(stage_index, len(stages)-1):
    prob_advancement[stages[i]+' to '+stages[i+1]] = st.sidebar.slider(f'{stages[i]} → {stages[i+1]}', 0, 100, 50, step=5)
prob_advancement['Series B to Series C'] = st.sidebar.slider('Series B → Series C', 0, 100, 40, step=5)
prob_advancement['Series C to IPO'] = st.sidebar.slider('Series C → IPO', 0, 100, 20, step=5)

st.sidebar.header('Dilution per Round (%)')
dilution = {}
for i in range(stage_index, len(stages)-1):
    dilution[stages[i]+' to '+stages[i+1]] = st.sidebar.slider(f'Dilution {stages[i]} → {stages[i+1]}', 0, 100, (15,30), step=5)
dilution['Series B to Series C'] = st.sidebar.slider('Dilution Series B → Series C', 0, 100, (15,25), step=5)
dilution['Series C to IPO'] = st.sidebar.slider('Dilution Series C → IPO', 0, 100, (10,20), step=5)

st.sidebar.header('Exit Valuations if No Further Progression ($MM)')
exit_valuations = {}
for stage in valid_stages + ['Series C', 'IPO']:
    if stage == 'Series C':
        exit_valuations[stage] = st.sidebar.slider(f'Exit Valuation at {stage}', 50, 300, (75, 150), step=5)
    elif stage == 'IPO':
        exit_valuations[stage] = st.sidebar.slider(f'Exit Valuation at {stage}', 100, 1000, (200, 500), step=10)
    else:
        exit_valuations[stage] = st.sidebar.slider(f'Exit Valuation at {stage}', 1, 200, (5, 20), step=1)

num_simulations = st.sidebar.slider('Number of Simulations', 1, 50, 20)

# Simulation function
def simulate_portfolio():
    investments = []

    for stage in valid_stages:
        allocation_amount = (stage_allocations[stage] / 100) * fund_size
        deployed_in_stage = 0

        while deployed_in_stage < allocation_amount:
            valuation = np.random.uniform(*valuations[stage])
            check_size = np.random.uniform(*check_sizes[stage])
            check_size = min(check_size, allocation_amount - deployed_in_stage)
            deployed_in_stage += check_size
            equity = check_size / valuation

            investment = {'Entry Stage': stage, 'Entry Amount': check_size}
            current_stage = stage

            stages_sequence = stages[stages.index(stage):] + ['Series C', 'IPO']
            for i, next_stage in enumerate(stages_sequence[1:], start=stages.index(stage)):
                key = stages_sequence[i-1] + ' to ' + next_stage
                if np.random.rand() * 100 <= prob_advancement.get(key, 0):
                    dilution_pct = np.random.uniform(*dilution.get(key, (0,0))) / 100
                    equity *= (1 - dilution_pct)
                    current_stage = next_stage
                else:
                    break

            exit_valuation = np.random.uniform(*exit_valuations[current_stage])
            investment.update({'Exit Stage': current_stage, 'Exit Amount': equity * exit_valuation})
            investments.append(investment)

    return pd.DataFrame(investments)

# Run simulations
all_sim_results = [simulate_portfolio() for _ in range(num_simulations)]
paid_in = [res['Entry Amount'].sum() for res in all_sim_results]
distributions = [res['Exit Amount'].sum() for res in all_sim_results]
moics = [d/p for d,p in zip(distributions, paid_in)]
irrs = [(moic ** (1/5) - 1)*100 for moic in moics]

# Display summary statistics
st.subheader("Simulation Summary Statistics")
cols = st.columns(5)
for col, metric, val in zip(cols, ["Paid-in", "Distributed", "MOIC", "IRR %", "# Investments"],
    [np.mean(paid_in), np.mean(distributions), np.mean(moics), np.mean(irrs), np.mean([len(r) for r in all_sim_results])]):
    col.metric(f"Avg. {metric}", f"{val:,.2f}")

# Visualization
st.subheader("Distribution of Fund MOIC")
fig, ax = plt.subplots(figsize=(8,4), dpi=120)
sns.histplot(moics, bins=15, kde=True, ax=ax)
st.pyplot(fig)

# Stacked Bar Chart of Entry vs. Exit
st.subheader("Entry Capital vs. Exit Value per Investment (Sample Simulation)")
sample_sim = all_sim_results[0].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

# Compute gain/loss per investment
exit_minus_entry = sample_sim['Exit Amount'] - sample_sim['Entry Amount']

ax.bar(sample_sim.index, sample_sim['Entry Amount'], label='Initial Investment', color='skyblue')
ax.bar(sample_sim.index, exit_minus_entry, bottom=sample_sim['Entry Amount'], 
       label='Gain / Loss', color='seagreen', alpha=0.7)

ax.set_xlabel('Investment #')
ax.set_ylabel('Value ($MM)')
ax.set_title('Stacked Entry Capital and Exit Value per Investment')
ax.legend()
st.pyplot(fig)


# Investment Schedule
st.subheader("Sample Simulation Investments")
st.dataframe(all_sim_results[0])

