import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

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

data = pd.read_csv('kidney_disease_cleaned1.csv')  # Path to your CKD dataset
data[['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']] = data[['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']].fillna('unknown')

st.sidebar.image('image.png')
page = st.sidebar.selectbox("Select a Page", ["Key Feature Indicators", "Numerical Feature Comparison", "Categorical Feature Comparison", "Prediction"])

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
        #st.pyplot(fig2)

        total_counts = filtered_data.groupby(age_groups).size()

        # CKD Positive count in each age group
        ckd_positive_counts = filtered_data[filtered_data['classification'] == 'CKD Possitve'].groupby(
            age_groups).size()

        # Combine the counts into a DataFrame for easier plotting and display
        counts_df = pd.DataFrame({
            'Total Count': total_counts,
            'CKD Positive Count': ckd_positive_counts
        })  # Fill NaNs with 0 for age groups with no CKD positive cases

        counts_df = counts_df.reset_index()
        counts_df.rename(columns={'age': 'Age Group'}, inplace=True)

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot total counts in the background (pink)
        counts_df['Total Count'].plot(kind='bar', color='pink', ax=ax, width=0.8, label='Total Count')

        # Plot CKD positive counts in the foreground (blue)
        counts_df['CKD Positive Count'].plot(kind='bar', color='royalblue', ax=ax, width=0.4,
                                             label='CKD Positive Count')

        # Adding labels above bars
        for i in range(len(counts_df)):
            # Positioning the total count label (pink) slightly above the bar
            ax.text(i, counts_df['Total Count'].iloc[i] + 0.5, int(counts_df['Total Count'].iloc[i]),
                    ha='center', va='bottom', color='pink', fontweight='bold')
            # Positioning the CKD positive count label (blue) slightly above the bar
            ax.text(i, counts_df['CKD Positive Count'].iloc[i] + 0.5, int(counts_df['CKD Positive Count'].iloc[i]),
                    ha='center', va='bottom', color='royalblue', fontweight='bold')

        # Chart formatting
        ax.set_title("CKD Positive Cases by Age Range")
        ax.set_xlabel("Age Range")
        ax.set_ylabel("Count")
        ax.legend()

        # Display the chart in Streamlit
        st.pyplot(fig)

        # Display the counts as a table in Streamlit
        #st.write("### Age Group Counts")
        #st.dataframe(counts_df)


elif page == "Numerical Feature Comparison":
    st.title("Comparative Analysis of Key Health Indicators by CKD Classification")

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
    st.title("CKD Risk Factors: Albumin, Suger Level and Specific Gravity")

    col1, col2 = st.columns(2)

    with col1:
        condition = (filtered_data['sg'] < 1.015) & (filtered_data['al'] == 0) & (filtered_data['su'] == 0)

        filtered_counts = filtered_data[condition]['classification'].value_counts()
        filtered_percentages = round((filtered_counts / filtered_data['classification'].value_counts().loc[
            'CKD Possitve']) * 100, 2)
        st.subheader("CKD Classification Analysis for Cases with AL = 0, SU = 0, and SG > 1.015")
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

elif page == "Prediction":
    df = pd.read_csv(
        'kidney_disease_cleaned3.csv')
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    # Feature and target variables
    X_train = train.drop('classification', axis=1)
    y_train = train['classification']
    X_test = test.drop('classification', axis=1)
    y_test = test['classification']

    # Preprocess data: Separate categorical and numerical features
    cat = []
    num = []

    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            cat.append(col)
        else:
            num.append(col)

    # Encode categorical variables
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoded_data = enc.fit_transform(X_train[cat])
    encoded_df = pd.DataFrame(encoded_data, columns=cat)
    X_train = pd.concat([X_train[num], encoded_df], axis=1)

    encoded_data2 = enc.transform(X_test[cat])
    encoded_df2 = pd.DataFrame(encoded_data2, columns=cat)
    X_test = pd.concat([X_test[num], encoded_df2], axis=1)

    # Store column names before scaling
    cols = X_train.columns

    # Normalize data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=cols)

    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=cols)

    # Label encode target variable
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Define RandomForest model
    model = RandomForestClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Streamlit interface
    st.title("Kidney Disease Prediction")
    st.write("### Enter Patient Data for Prediction")

    # Input fields
    age = st.number_input("Age", min_value=0, max_value=200)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=500)
    sg = st.number_input("Specific Gravity", min_value=1.00, max_value=2.00)
    al = st.number_input("Albumin", min_value=0, max_value=10)
    su = st.number_input("Sugar Level", min_value=0, max_value=10)
    bgr = st.number_input("Blood Glucose", min_value=0, max_value=2000)
    bu = st.number_input("Blood Urea", min_value=0, max_value=2000)
    sc = st.number_input("Serum Creatine", min_value=0.0, max_value=1000.0)
    sod = st.number_input("Sodium", min_value=0, max_value=2000)
    pot = st.number_input("Potassium", min_value=0.0, max_value=1000.0)
    hemo = st.number_input("Hemoglobin", min_value=0.0, max_value=200.0)
    pcv = st.number_input("Packed Cell Volume", min_value=0.0, max_value=100.0)
    wc = st.number_input("White Blood Cell Count", min_value=0.0, max_value=100000.0)
    rc = st.number_input("Red Blood Cell Count", min_value=0.0, max_value=100.0)

    # Categorical fields as dropdowns
    rbc = st.selectbox("Red Blood Cell Clusters", ['normal', 'abnormal', 'unknown'])
    pc = st.selectbox("Pus Cell", ['normal', 'abnormal', 'unknown'])
    pcc = st.selectbox("Pus Cell Clumps", ['normal', 'abnormal', 'unknown'])
    ba = st.selectbox("Bacteria", ['not present', 'present', 'unknown'])
    htn = st.selectbox("Hypertension", ['yes', 'no', 'unknown'])
    dm = st.selectbox("Diabetes Mellitus", ['yes', 'no', 'unknown'])
    cad = st.selectbox("Coronary Artery Disease", ['yes', 'no', 'unknown'])
    appet = st.selectbox("Appetite", ['good', 'poor', 'unknown'])
    pe = st.selectbox("Pedal Edema", ['yes', 'no', 'unknown'])
    ane = st.selectbox("Anemia", ['yes', 'no', 'unknown'])


    # Define the prediction function
    def predict(model):
        data = [age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, rbc, pc, pcc, ba, htn, dm, cad, appet,
                pe, ane]
        data = np.array(data).reshape(1, -1)
        df_input = pd.DataFrame(data, columns=X_train.columns)

        # Preprocess the input data (similar to training data)
        encoded_input = enc.transform(df_input[cat])
        encoded_df_input = pd.DataFrame(encoded_input, columns=cat)
        df_input = pd.concat([df_input[num], encoded_df_input], axis=1)
        df_input = scaler.transform(df_input)
        df_input = pd.DataFrame(df_input, columns=cols)

        # Make prediction
        prediction = model.predict(df_input)
        prediction_label = le.inverse_transform(prediction)[0]

        return prediction_label


    # Button to trigger prediction
    if st.button("Predict"):
        prediction_result = predict(model)
        st.write(f"The prediction result is: {prediction_result}")
