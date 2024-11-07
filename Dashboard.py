import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.markdown(
    """
    <style>
    /* Set background color to light blue and grey mix */
    .stApp {
        background-color: #EAF2F8;  /* Light Blue */
        background-image: linear-gradient(135deg, #EAF2F8 40%, #BDC3C7 100%);  /* Light blue to grey gradient */
    }

    /* Set sidebar background color to dark grey */
    .css-1d391kg {
        background-color: #2E2E2E !important;  /* Dark Grey */
    }

    /* Style for sidebar titles and texts (white) */
    .css-1vbd788, .css-j7qwjs, .css-1vencpc {
        color: white !important;  /* Make sidebar text white */
    }

    /* Style for the main section titles (dark grey) */
    .css-10trblm {
        color: #2C3E50 !important;  /* Darker grey for the main section titles */
    }

    /* Style for KPI metric numbers */
    .css-2trqyj {
        font-size: 2rem !important;  /* Increase KPI font size */
        color: #34495E !important;   /* Dark blue-grey */
    }
    </style>
    """,
    unsafe_allow_html=True
)

data = pd.read_csv(r'C:\Users\Rukmal\PycharmProjects\pythonProject\Internship\Project_CKD\kidney_disease_cleaned1.csv')  # Path to your CKD dataset
data[['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']] = data[['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']].fillna('unknown')

st.sidebar.image(r'C:\Users\Rukmal\PycharmProjects\pythonProject\Internship\Project_CKD\image.png')
page = st.sidebar.selectbox("Select a Page", ["Key Feature Indicators", "Numerical Feature Comparison", "Categorical Feature Comparison"])

with st.sidebar.form("Options"):  # Change from 'form1' to 'st.sidebar.form'
    st.header("Filters")

    rbc_select = data['rbc'].unique()
    rbc_filter = st.multiselect("Red Blood Cells in Urine", rbc_select, default=rbc_select)

    pc_select = data['pc'].unique()
    pc_filter = st.multiselect("Puss Cells", pc_select, default=pc_select)

    pcc_select = data['pcc'].unique()
    pcc_filter = st.multiselect("Puss Cell Cluster", pcc_select, default=pcc_select)

    ba_select = data['ba'].unique()
    ba_filter = st.multiselect("Bacteria", ba_select, default=ba_select)

    htn_select = data['htn'].unique()
    htn_filter = st.multiselect("Hypertension", htn_select, default=htn_select)

    dm_select = data['dm'].unique()
    dm_filter = st.multiselect("Diabetes Mellitus", dm_select, default=dm_select)

    cad_select = data['cad'].unique()
    cad_filter = st.multiselect("Chronic Artery Disease", cad_select, default=cad_select)

    appet_select = data['appet'].unique()
    appet_filter = st.multiselect("Appetite", appet_select, default=appet_select)

    pe_select = data['pe'].unique()
    pe_filter = st.multiselect("Pedal Edema", pe_select, default=pe_select)

    ane_select = data['ane'].unique()
    ane_filter = st.multiselect("Anemia", ane_select, default=ane_select)

    form_submit = st.form_submit_button("Apply")

if form_submit:
    filtered_data = data[
        (data['htn'].isin(htn_filter)) &
        (data['dm'].isin(dm_filter)) &
        (data['ane'].isin(ane_filter)) &
        (data['pe'].isin(pe_filter)) &
        (data['rbc'].isin(rbc_filter)) &
        (data['appet'].isin(appet_filter)) &
        (data['pc'].isin(pc_filter)) &
        (data['pcc'].isin(pcc_filter)) &
        (data['ba'].isin(ba_filter)) &
        (data['cad'].isin(cad_filter))
        ]
else:
    filtered_data = data

if page == "Key Feature Indicators":
    st.title("Key Feature Indicators")
    st.subheader("CKD Distribution")

    col1, col2 = st.columns(2)

    # KPI 1: CKD Positive/Negative Counts
    with col1:
        if not filtered_data.empty:
            ckd_counts = filtered_data['classification'].value_counts()
            st.metric("CKD Positive Cases", ckd_counts.get('CKD Possitve', 0))
            st.metric("CKD Negative Cases", ckd_counts.get('CKD Negettive', 0))
        else:
            st.write("No data available with the selected filters.")

        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.pie(ckd_counts, labels=ckd_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
        ax1.set_title("CKD Positive vs. CKD Negative Cases")
        st.pyplot(fig1)

    # KPI 2: Age Group Distribution of CKD Positive Cases
    with col2:
        age_groups = pd.cut(filtered_data['age'], bins=range(0, 101, 10))
        age_group_counts = filtered_data[filtered_data['classification'] == 'CKD Possitve'].groupby(age_groups).size()
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        age_group_counts.plot(kind='bar', color='royalblue', ax=ax2)
        ax2.set_title("CKD Positive Cases by Age Range")
        ax2.set_xlabel("Age Range")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)


elif page == "Numerical Feature Comparison":
    st.title("Numerical Feature Comparison: CKD Positive vs. Negative")

    col1, col2, col3 = st.columns([3,2,3])

    with col1:
        numeric_columns = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
        filtered_data[numeric_columns] = filtered_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
        ckd_group = filtered_data.groupby('classification')[numeric_columns].mean().round(0).reset_index()
        ckd_group = ckd_group.drop(columns=['id', 'age', 'sg', 'al', 'su'])
        ckd_group = ckd_group.set_index('classification')

        st.subheader("Chart 1")
        df_transposed1 = ckd_group[['bp', 'bgr', 'bu', 'sod', 'pcv']].T
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        df_transposed1.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_xlabel("Parameters")
        ax1.set_ylabel("Mean Value")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.legend(title="Classification")
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig1)

        # Select only relevant columns and transpose for better readabilityT
        st.dataframe(df_transposed1.style.format("{:.2f}"))

    # Second chart: Only for 'wc' parameter
    with col2:
        st.subheader("Chart 2")
        df_transposed2 = ckd_group[['wc']].T
        custom_colors = ['#4682B4', '#FF6347']
        fig2, ax2 = plt.subplots(figsize=(2, 4))
        df_transposed2.plot(kind='bar', ax=ax2, width=0.8, color=custom_colors)
        ax2.set_xlabel("Classification")
        ax2.set_ylabel("WC Level")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig2)

        st.dataframe(df_transposed2.style.format("{:.2f}"))

    with col3:
        st.subheader("Chart 3")
        df_transposed3 = ckd_group[['sc', 'pot', 'hemo', 'rc']].T
        fig3, ax3 = plt.subplots(figsize=(4, 4))
        df_transposed3.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_xlabel("Parameters")
        ax3.set_ylabel("Mean Value")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.legend(title="Classification")
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig3)

        st.dataframe(df_transposed3.style.format("{:.2f}"))


elif page == "Categorical Feature Comparison":
    st.title("Analysis of Albumin, Suger Level and Specific Gravity")

    col1, col2 = st.columns(2)

    with col1:
        condition = (filtered_data['sg'] < 1.015) & (filtered_data['al'] == 0) & (filtered_data['su'] == 0)

        filtered_counts = filtered_data[condition]['classification'].value_counts()
        filtered_percentages = round((filtered_counts / filtered_data['classification'].value_counts().loc[
            'CKD Possitve']) * 100, 2)
        st.subheader("CKD Classification Where, AL = 0, SU = 0, and SG > 1.015")
        st.metric("Count", filtered_counts)
        st.metric("Percentage (%)", filtered_percentages)
        # st.dataframe(result_table)


        al_ckd = pd.crosstab(filtered_data['al'], filtered_data['classification'])

        plt.figure(figsize=(10, 6))
        al_ckd.plot(kind='bar', color=['#66b3ff', '#ff9999'], width=0.8)

        plt.title("CKD by Albumin Content")
        plt.xlabel("Albumin Content Levels")
        plt.ylabel("Count")
        plt.legend(title="Classification", labels=["CKD Negative", "CKD Positive"])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(plt)



    with col2:
        su_ckd = pd.crosstab(filtered_data['su'], filtered_data['classification'])

        plt.figure(figsize=(10, 6))
        su_ckd.plot(kind='bar', color=['#66b3ff', '#ff9999'], width=0.8)

        plt.title("CKD by Suger Level")
        plt.xlabel("Suger Levels")
        plt.ylabel("Count")
        plt.legend(title="Classification", labels=["CKD Negative", "CKD Positive"])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(plt)

        sg_ckd = pd.crosstab(filtered_data['sg'], filtered_data['classification'])

        plt.figure(figsize=(10, 6))
        sg_ckd.plot(kind='bar', color=['#66b3ff', '#ff9999'], width=0.8)

        plt.title("CKD by Specific Gravity")
        plt.xlabel("Specific Gravity Levels")
        plt.ylabel("Count")
        plt.legend(title="Classification", labels=["CKD Negative", "CKD Positive"])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(plt)


        # Combine the counts and percentages into a DataFrame for display
        #result_table = pd.DataFrame({
            #"Count": filtered_counts,
            #"Percentage (%)": filtered_percentages
        #}).reset_index()

        # Rename columns for clarity
        #result_table = result_table.rename(columns={'index': 'Classification'})

        # Display the table in Streamlit
