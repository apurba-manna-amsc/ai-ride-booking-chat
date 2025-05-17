import streamlit as st
from general import receive_message_from_caller, ai_response, reset_session,get_last_5_records

st.set_page_config(page_title="AI Ride Booking Chat", page_icon="🚖")

# Title
st.title("🚖 AI Ride Booking System")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for session management
with st.sidebar:
    st.header("Session Management")
    if st.button("Reset Session"):
        if 'reset_session' in globals():
            reset_session()  # Custom logic from general.py
        st.session_state.chat_history = []  # Clear chat from Streamlit session
        st.success("Session reset successfully!")

    st.header("Recent Bookings")
    if st.button("Refresh Bookings"):
        try:
            recent_records = get_last_5_records()
            if recent_records:
                # Display bookings in a more structured way
                st.subheader(f"Found {len(recent_records)} recent bookings:")
                
                # Create a better formatted dataframe
                booking_data = []
                for record in recent_records:
                    # Convert record to appropriate format
                    booking_data.append({
                        "ID": record[0],
                        "Pickup": record[1],
                        "Dropoff": record[2],
                        "Type": record[3],
                        "Scheduled": record[4].strftime("%Y-%m-%d %H:%M") if record[4] else "N/A",
                        "Booked on": record[5].strftime("%Y-%m-%d %H:%M") if record[5] else "N/A"
                    })
                
                # Display as a dataframe
                st.dataframe(booking_data)
            else:
                st.info("No recent bookings found.")
        except Exception as e:
            st.error(f"Failed to load recent bookings: {str(e)}")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Type your message or 'exit' to quit...")

if user_input:
    st.session_state.chat_history.append(("You", user_input))

    if user_input.lower() == 'exit':
        st.session_state.chat_history.append(("AI", "Thank you for using our service. Goodbye!"))
    else:
        try:
            receive_message_from_caller(user_input)
            if ai_response:
                st.session_state.chat_history.append(("AI", ai_response[-1]))
        except Exception as e:
            st.session_state.chat_history.append(("AI", f"Error: {str(e)}"))

# Display chat
for speaker, message in st.session_state.chat_history:
    with st.chat_message("user" if speaker == "You" else "assistant"):
        st.markdown(message)
