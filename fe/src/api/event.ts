import client from "./client";

export interface EventItem {
  id: number;
  camera_id: string;
  detected_at: string;
  anomaly_type: string;
  confidence: number;
  status: "UNREVIEWED" | "REVIEWED" | "FALSE_POSITIVE";
  created_at: string;
}

export interface EventDetail extends EventItem {
  description: string | null;
  video_url: string | null;
}

export interface EventListResponse {
  items: EventItem[];
  total: number;
  page: number;
  size: number;
}

export async function listEvents(params?: {
  page?: number;
  size?: number;
  status?: string;
  camera_id?: string;
}): Promise<EventListResponse> {
  const res = await client.get("/api/events", { params });
  return res.data.data;
}

export async function getEvent(id: number): Promise<EventDetail> {
  const res = await client.get(`/api/events/${id}`);
  return res.data.data;
}

export async function updateEventStatus(
  id: number,
  status: "REVIEWED" | "FALSE_POSITIVE"
): Promise<EventItem> {
  const res = await client.patch(`/api/events/${id}/status`, { status });
  return res.data.data;
}
