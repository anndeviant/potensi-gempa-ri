export interface PredictionResponse {
    status: string;
    data: {
        hazard_level: string;
        lat: number;
        lng: number;
    };
    // message?: string; // Untuk menangani pesan error jika ada
}

export interface PredictionM5RadiusResponse {
    status: string;
    data: {
        lat: number;
        lng: number;
        radius_km: number;
        nearest_region: string;
        distance_km: number;
        n_regions_in_radius: number;
        estimated_total_m5_in_radius: number;
        open_month: string;
        target_month: string;
        model_last_observed_month: string;
    };
}

const API_BASE_URL =
    process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "https://peta-potensi-gempa-ri-api-v1.vercel.app";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY;

const buildHeaders = (): HeadersInit => {
    if (!API_KEY) {
        throw new Error("NEXT_PUBLIC_API_KEY belum diset.");
    }

    return {
        Accept: "application/json",
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
    };
};

export const fetchHazardPrediction = async (lat: number, lng: number): Promise<PredictionResponse | null> => {
    try {
        const url = `${API_BASE_URL}/predict?lat=${lat}&lng=${lng}`;

        const response = await fetch(url, {
            method: 'GET',
            headers: buildHeaders(),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result: PredictionResponse = await response.json();
        console.log("Prediksi berhasil diterima:", result);
        return result;
    } catch (error) {
        console.error("Gagal mengambil prediksi dari Flask:", error);
        return null;
    }
};

export const fetchM5RadiusPrediction = async (
    lat: number,
    lng: number,
    radiusKm: number
): Promise<PredictionM5RadiusResponse | null> => {
    try {
        const url = `${API_BASE_URL}/predict-m5-radius?lat=${lat}&lng=${lng}&radius_km=${radiusKm}`;

        const response = await fetch(url, {
            method: 'GET',
            headers: buildHeaders(),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result: PredictionM5RadiusResponse = await response.json();
        console.log("Prediksi M>=5 radius berhasil diterima:", result);
        return result;
    } catch (error) {
        console.error("Gagal mengambil prediksi M>=5 radius dari Flask:", error);
        return null;
    }
};
