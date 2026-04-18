import client from "./client";

export interface Camera {
  id: number;
  cameraId: string;
  name: string;
  location: string;
  status: string;
  active: boolean;
}

export interface CameraDetail extends Camera {
  streamKey: string;
  description: string;
  createdAt: string;
  updatedAt: string;
}

export interface StreamStatus {
  cameraId: string;
  cameraStatus: string;
  currentSession: {
    sessionId: number;
    status: string;
    hlsUrl: string;
    startedAt: string;
    stoppedAt: string | null;
  } | null;
}

export const getCameras = async (): Promise<Camera[]> => {
  const res = await client.get("/api/cameras");
  return res.data.data;
};

export const getCameraDetail = async (cameraId: string): Promise<CameraDetail> => {
  const res = await client.get(`/api/cameras/${cameraId}`);
  return res.data.data;
};

export const createCamera = async (data: {
  cameraId: string;
  name: string;
  location: string;
  streamKey: string;
  description?: string;
}) => {
  const res = await client.post("/api/cameras", data);
  return res.data.data;
};

export const deleteCamera = async (cameraId: string) => {
  const res = await client.delete(`/api/cameras/${cameraId}`);
  return res.data.data;
};

export const startStream = async (cameraId: string) => {
  const res = await client.post(`/api/cameras/${cameraId}/stream/start`, {});
  return res.data.data;
};

export const stopStream = async (cameraId: string) => {
  const res = await client.post(`/api/cameras/${cameraId}/stream/stop`, {});
  return res.data.data;
};

export const getStreamStatus = async (cameraId: string): Promise<StreamStatus> => {
  const res = await client.get(`/api/cameras/${cameraId}/stream`);
  return res.data.data;
};
