import client from "./client";

export interface User {
  id: number;
  email: string;
  name: string;
  role: string;
  status: string;
  createdAt: string;
}

export const getUsers = async (): Promise<User[]> => {
  const res = await client.get("/api/users");
  return res.data.data;
};

export const approveUser = async (userId: number) => {
  const res = await client.patch(`/api/users/${userId}/approve`);
  return res.data.data;
};

export const deleteUser = async (userId: number) => {
  const res = await client.delete(`/api/users/${userId}`);
  return res.data.data;
};

export const assignCamera = async (cameraId: string, userId: number) => {
  const res = await client.post(`/api/users/cameras/${cameraId}/assign`, { user_id: userId });
  return res.data.data;
};

export const unassignCamera = async (cameraId: string, userId: number) => {
  const res = await client.delete(`/api/users/cameras/${cameraId}/assign/${userId}`);
  return res.data.data;
};

export const getUserCameras = async (userId: number): Promise<{ cameraId: string; name: string }[]> => {
  const res = await client.get(`/api/users/${userId}/cameras`);
  return res.data.data;
};
