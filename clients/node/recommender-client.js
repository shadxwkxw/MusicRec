/**
 * Node.js Client for Music Recommender ML Service
 *
 * Drop this into your Spotify-clone backend to call the
 * FastAPI recommendation service.
 *
 * Usage:
 *   const recommender = require('./recommender-client');
 *
 *   // Upload a track
 *   const track = await recommender.uploadTrack('/path/to/song.mp3', 'My Song', 'Artist');
 *
 *   // Get recommendations
 *   const recs = await recommender.getRecommendations(track.id, 10);
 */

const fs = require("fs");
const path = require("path");

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";

/**
 * Upload an audio track and extract features.
 */
async function uploadTrack(filePath, title, artist = "Unknown") {
  const formData = new FormData();
  const fileBuffer = fs.readFileSync(filePath);
  const blob = new Blob([fileBuffer]);
  formData.append("file", blob, path.basename(filePath));
  formData.append("title", title);
  formData.append("artist", artist);

  const res = await fetch(`${ML_SERVICE_URL}/tracks/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(`Upload failed: ${JSON.stringify(err)}`);
  }

  return res.json();
}

/**
 * Get content-based recommendations for a track.
 */
async function getRecommendations(trackId, limit = 10, useLikes = true) {
  const params = new URLSearchParams({
    limit: limit.toString(),
    use_likes: useLikes.toString(),
  });

  const res = await fetch(
    `${ML_SERVICE_URL}/recommendations/${trackId}?${params}`
  );

  if (!res.ok) {
    const err = await res.json();
    throw new Error(`Recommendations failed: ${JSON.stringify(err)}`);
  }

  return res.json();
}

/**
 * Get personalized recommendations for a user.
 */
async function getUserRecommendations(userId, limit = 10) {
  const res = await fetch(
    `${ML_SERVICE_URL}/recommendations/user/${userId}?limit=${limit}`
  );

  if (!res.ok) {
    const err = await res.json();
    throw new Error(`User recommendations failed: ${JSON.stringify(err)}`);
  }

  return res.json();
}

/**
 * Record a user like (sync likes from your main DB to ML service).
 */
async function syncLike(userId, trackId) {
  const res = await fetch(`${ML_SERVICE_URL}/likes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, track_id: trackId }),
  });

  return res.json();
}

/**
 * Trigger AutoML optimization (call periodically or after adding many tracks).
 */
async function triggerAutoML() {
  const res = await fetch(`${ML_SERVICE_URL}/automl/train`, {
    method: "POST",
  });
  return res.json();
}

/**
 * Check AutoML training status.
 */
async function getAutoMLStatus() {
  const res = await fetch(`${ML_SERVICE_URL}/automl/status`);
  return res.json();
}

/**
 * Rebuild the search index (after bulk imports).
 */
async function rebuildIndex() {
  const res = await fetch(`${ML_SERVICE_URL}/index/rebuild`, {
    method: "POST",
  });
  return res.json();
}

module.exports = {
  uploadTrack,
  getRecommendations,
  getUserRecommendations,
  syncLike,
  triggerAutoML,
  getAutoMLStatus,
  rebuildIndex,
};
