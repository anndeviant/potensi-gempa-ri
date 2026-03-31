import Papa from 'papaparse';



export interface Earthquake {
    eventID: string;
    datetime: string;
    latitude: number;
    longitude: number;
    magnitude: number;
    location: string;
    mag_type: number;
    depth: number;
    phasecount: number;
    azimuth_gap: number;
    dist?: number; // kita tambahkan untuk menyimpan hasil hitung jarak
}

export interface EQGrid {
    grid_id: string;
    lat: number;
    lon: number;
    freq: number;
    max_mag: number;
    avg_mag: number;
    avg_depth: number;
    gempa_in_radius_50: number;
    density: number;
}

// Fungsi Rumus Haversine
const calculateDistance = (lat1: number, lon1: number, lat2: number, lon2: number) => {
    const R = 6371; // Radius bumi dalam kilometer
    const dLat = (lat2 - lat1) * (Math.PI / 180);
    const dLon = (lon2 - lon1) * (Math.PI / 180);
    const a =
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(lat1 * (Math.PI / 180)) * Math.cos(lat2 * (Math.PI / 180)) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c; // Jarak dalam KM
};

export const findNearestEarthquake = async (userLat: number, userLng: number) => {
    const earthquakes = await membacaFileCSVGempa('/data/katalog_gempa_v2_cleaned.csv');
    if (earthquakes != null) {
        const earthquakesWithDistance = earthquakes.map((eq) => ({
            ...eq,
            dist: calculateDistance(userLat, userLng, eq.latitude, eq.longitude)
        }));
        const sorted = earthquakesWithDistance.sort((a, b) => (a.dist || 0) - (b.dist || 0));

        return sorted[0];
    } else {
        return null;
    }
};

const membacaFileCSVGempa = async (filePath: string): Promise<Earthquake[] | null> => {
    try {
        const response = await fetch(filePath);
        const csvText = await response.text();

        const parsed = Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
        });
        console.log("Data CSV berhasil dibaca:", parsed.data[0]);
        return parsed.data as Earthquake[];
    } catch (error) {
        console.error("Gagal membaca file CSV:", error);
        return null;
    }
};

const membacaFileCSVGempaV2 = async (filePath: string): Promise<EQGrid[] | null> => {
    try {
        const response = await fetch(filePath);
        const csvText = await response.text();

        const parsed = Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
        });
        console.log("Data CSV V2 berhasil dibaca:", parsed.data[0]);
        return parsed.data as EQGrid[];
    } catch (error) {
        console.error("Gagal membaca file CSV V2:", error);
        return null;
    }   
};


export const getAreaStatistics = async (
    userLat: number,
    userLng: number,
    radiusKm: number = 50 // Default radius 50km
): Promise<{ 
    frequency: number;
    maxMagnitude: string | number;
    avgMagnitude: string | number;
    avgDepth: string | number;
} | null> => {

    const earthquakes = await membacaFileCSVGempa('/data/katalog_gempa_v2_cleaned.csv');
    if (!earthquakes) return null;

    const nearbyEarthquakes = earthquakes.filter((eq) => {
        const distance = calculateDistance(userLat, userLng, eq.latitude, eq.longitude);
        return distance <= radiusKm;
    });
    const total = nearbyEarthquakes.length;
    if (total === 0) {
        return {
            frequency: 0,
            maxMagnitude: 0,
            avgMagnitude: 0,
            avgDepth: 0,
        };
    }
    const stats = nearbyEarthquakes.reduce(
        (acc, cur) => {
            // Cari Magnitudo Terbesar
            if (cur.magnitude > acc.maxMag) acc.maxMag = cur.magnitude;
            // Jumlahkan untuk rata-rata
            acc.sumMag += cur.magnitude;
            acc.sumDepth += cur.depth;
            return acc;
        },
        { maxMag: 0, sumMag: 0, sumDepth: 0 }
    );

    return {
        frequency: total,
        maxMagnitude: stats.maxMag.toFixed(2),
        avgMagnitude: (stats.sumMag / total).toFixed(2),
        avgDepth: (stats.sumDepth / total).toFixed(2),
    };
};