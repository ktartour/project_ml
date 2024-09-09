import streamlit as st



def print_images():
    st.write("This brief application aim to develop three methods of machine learning")
    st.write("Firstly, with the **regression**, you can train a model to estimate the progression of diabetes from biological parameters")
    st.write("Secondly, with the **classifcation**, you can train models to classify wines from their physico-chemical parameters")
    st.write("Thirdly, with the **NailsDetection** a reviously trained model is used to recognize nails from the picture of your choice")
    # Load images
    urla= "https://www.inneance.fr/wp-content/uploads/machine_learning-1080x569.jpg"
    urlb ="https://i0.wp.com/eos.org/wp-content/uploads/2020/07/data-streams-eos-august.png?w=820&ssl=1"
    urlc = "https://d3caycb064h6u1.cloudfront.net/wp-content/uploads/2021/06/Machine_Learning.jpg"

    #st.image(urla)
    st.sidebar.image(urlb)
    st.image(urlc)
    # URLs of the images
    url1 = 'https://cdn.prod.website-files.com/5f6bc60e665f54545a1e52a5/612ceede647190109abb0541_full-logo-p-500.png'
    url2 = 'https://miro.medium.com/v2/resize:fit:1400/0*5yINw4AB2CojpC0X.png'  # Replace with the actual URL of the second image
    url3 = "data:image/png;base64,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"
    # Create three columns
    col1, col2, col3 = st.columns([1, 1, 1])  # Equal width columns

    # Display images in the columns
    with col1:
        st.image(url3, caption=None, use_column_width=True)

    with col3:
        st.image(url1, caption=None, use_column_width=True)

    with col2:
        st.image(url2, caption=None, use_column_width=True)



    return None