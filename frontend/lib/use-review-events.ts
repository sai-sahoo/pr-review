import { useEffect, useReducer } from "react";

import { initialState, reviewReducer, type ReviewState } from "./review-state";
import { API_URL, EVENT_TYPES, type ReviewEvent } from "./types";

// Opens the review's event stream and keeps the reducer's state up to date.
export function useReviewEvents(id: string): ReviewState {
  const [state, dispatch] = useReducer(reviewReducer, initialState);

  useEffect(() => {
    // The server replays every event from the start on a new connection, so
    // start from an empty state. In dev, React runs this effect twice on
    // purpose (mount, cleanup, mount) and the second run must not add the
    // replayed events on top of the first run's.
    dispatch({ type: "reset" });

    const source = new EventSource(`${API_URL}/reviews/${id}/events`);

    // Our server names every event ("event: node", ...). Named events only
    // reach listeners registered for that name; `onmessage` would see none.
    for (const type of EVENT_TYPES) {
      source.addEventListener(type, (message) => {
        dispatch(JSON.parse(message.data) as ReviewEvent);
        // The server ends the stream after done/failed. Left open, EventSource
        // would see that as a dropped connection and reconnect every few seconds.
        if (type === "done" || type === "failed") source.close();
      });
    }

    source.onerror = () => {
      // CONNECTING: a dropped connection; the browser is already retrying,
      //   sending Last-Event-ID so the server resumes where we left off.
      // CLOSED: the browser gave up (the server is down or answered 404).
      if (source.readyState === EventSource.CLOSED) {
        dispatch({ type: "failed", error: "Lost the connection to the API. Is the backend running?" });
      }
    };

    return () => source.close(); // leaving the page: stop listening
  }, [id]);

  return state;
}
