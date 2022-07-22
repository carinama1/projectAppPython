const startSession = () => {
  const session_name = document.getElementById("session_name").value;
  const current_session = localStorage.getItem("current_session");
  localStorage.setItem(
    "current_session",
    JSON.stringify({ session_id, session_name })
  );
  var origin = window.location.origin;
  window.location.replace(`${origin}/live`);
};
