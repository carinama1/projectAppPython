const videoDone = async () => {
  const session_name = JSON.parse(
    localStorage.getItem("current_session")
  ).session_name;
  await $.get(`/generate?session_name=${session_name}`);
  //   $("#live-feed").remove();
  const origin = window.location.origin;

  window.location.replace(`${origin}/results`);
};

const test = () => {
  const session_name = JSON.parse(
    localStorage.getItem("current_session")
  ).session_name;
  $.get(`/test?session_name=${session_name}`);
};
