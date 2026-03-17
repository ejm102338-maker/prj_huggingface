import streamlit as st

st.title("Session state의 중요성")
st.markdown("## Session State 사용")

if "count" not in st.session_state:
    st.session_state["count"] = 0

st.markdown("## Session State")
st.write(st.session_state)

# button
mybutton = st.button(
    label="버튼",
    key="mybutton",
    width="content" # content,stretch
)


# st.markdown("## Session State 구분없이 구현")
# count = 0
# st.markdown(f"count 초기 설정 : {count}")

# # button
# mybutton = st.button(
#     label="버튼"
# )

if mybutton:
    #count += 1
    st.session_state["count"] += 1
    #st.markdown(f"count = {count}")

if "count" in st.session_state:
    #st.session_state["count"] = count
    st.markdown(f"count : {st.session_state["count"]}")