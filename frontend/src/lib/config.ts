// Central place for the API base URL.
//
// In the Arena preview the frontend and backend are served from different
// origins, so we talk to the API on its own origin. In a normal local setup
// you can set VITE_API_URL (e.g. a deployed API) or leave the default.

declare global {
  // eslint-disable-next-line no-var
  var __API_BASE__: string | undefined
}

export const API_BASE: string =
  (typeof __API_BASE__ !== 'undefined' && __API_BASE__) ||
  import.meta.env.VITE_API_URL ||
  '/api'

export const DISCLAIMER =
  'DiaPredict AI is an educational and research prototype. It provides machine-learning-based risk estimates and is not a medical diagnosis. The results should not be used to make medical decisions. Always consult a qualified healthcare professional for medical advice.'
