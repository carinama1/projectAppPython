const getLocalStorage = () => {
  return localStorage.getItem("pass_sessions");
};

const set_currentSession = (current_session) => {
  localStorage.setItem("current_session", current_session);
};

const get_currentSession = () => {
  return localStorage.getItem("current_session");
};
