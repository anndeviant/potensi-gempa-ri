"use client";
import React, { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { Button } from "@/components/ui/button";
import { Earthquake } from "@/lib/operasiFileCSV";
import { findNearestEarthquake } from "@/lib/operasiFileCSV";
import formatTanggalIndo from "@/lib/formatTanggalIndo";
import { getAreaStatistics } from "@/lib/operasiFileCSV";
import {
  fetchHazardPrediction,
  fetchM5RadiusPrediction,
  PredictionM5RadiusResponse,
  PredictionResponse,
} from "@/lib/requestApi";

// Interface untuk State
interface Location {
  lat: number;
  lng: number;
}

interface MapProps {
  currentLocation: Location | null;
  clickedLocation: Location | null;
  setClickedLocation: (loc: Location) => void;
}

// Load Map tanpa SSR (Client Side Only)
const MapWithNoSSR = dynamic(() => import("../components/MapComponents"), {
  ssr: false,
  loading: () => (
    <div className=" h-full w-full flex items-center justify-center bg-gray-100 text-gray-500">
      Memuat Peta...
    </div>
  ),
});

export default function Home() {
  const [currentLocation, setCurrentLocation] = useState<Location | null>(null);
  const [clickedLocation, setClickedLocation] = useState<Location | null>(null);
  const [currentAddress, setCurrentAddress] = useState<string>("");
  const [clickedAddress, setClickedAddress] = useState<string>("");

  const [nearestEarthquake, setNearestEarthquake] = useState<Earthquake | null>(
    null,
  );
  const [nearestEquakeAddress, setNearestEarthquakeAddress] =
    useState<string>("");
  // Untuk menyimpan hasil prediksi model 1
  const [predictionModel1, setPredictionModel1] =
    useState<PredictionResponse | null>(null);
  const [predictionModel2, setPredictionModel2] =
    useState<PredictionM5RadiusResponse | null>(null);
  const [radiusKm, setRadiusKm] = useState<number>(50);

  const [areaStats, setAreaStats] = useState<{
    frequency: number;
    maxMagnitude: string | number;
    avgMagnitude: string | number;
    avgDepth: string | number;
  } | null>(null);

  // Ambil lokasi dari browser
  useEffect(() => {
    if (typeof window !== "undefined" && "geolocation" in navigator) {
      const options: PositionOptions = {
        enableHighAccuracy: true, // Meminta akurasi tertinggi (GPS)
        timeout: 10000, // Tunggu maksimal 10 detik
        maximumAge: 0, // Jangan gunakan lokasi cache lama
      };

      navigator.geolocation.getCurrentPosition(
        (position) => {
          console.log("Lokasi ditemukan:", position.coords);
          setCurrentLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          });
        },
        (error) => {
          // Handle error spesifik agar kamu tahu kenapa gagal di HP
          switch (error.code) {
            case error.PERMISSION_DENIED:
              alert(
                "Izin lokasi ditolak. Mohon aktifkan GPS di pengaturan browser untuk memudahkan penggunaan fitur Peta Potensi Gempa Indonesia.",
              );
              break;
            case error.POSITION_UNAVAILABLE:
              alert("Informasi lokasi tidak tersedia.");
              break;
            case error.TIMEOUT:
              alert("Waktu permintaan lokasi habis.");
              break;
          }
        },
        options,
      );
    }
  }, []);

  const getAddress = async (lat: number, lng: number) => {
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lng}`,
        {
          headers: {
            "Accept-Language": "id", // Agar hasil dalam Bahasa Indonesia
          },
        },
      );
      const data = await response.json();
      console.log("Data reverse geocode:", data);
      // Simpan alamat lengkap ke state
      return data.display_name; // Ini adalah nama lokasi lengkapnya
    } catch (error) {
      console.error("Gagal reverse geocode:", error);
      return "Lokasi tidak diketahui";
    }
  };

  useEffect(() => {
    if (clickedLocation) {
      getAddress(clickedLocation.lat, clickedLocation.lng).then((address) => {
        console.log("Alamat lokasi yang diklik:", address);
        setClickedAddress(address); // Simpan alamat lokasi yang diklik ke state
      });
    }
  }, [clickedLocation]);
  useEffect(() => {
    if (currentLocation) {
      getAddress(currentLocation.lat, currentLocation.lng).then((address) => {
        console.log("Alamat lokasi saat ini:", address);
        setCurrentAddress(address);
      });
    }
  }, [currentLocation]);

  const handleScrollToMain = () => {
    const mainSection = document.getElementById("main");
    if (mainSection) {
      mainSection.scrollIntoView({ behavior: "smooth" });
    }
  };

  const runAnalisis = useCallback(
    async (location: Location) => {
      console.log("Memulai analisis untuk lokasi:", location);
      if (location) {
        // Untuk penerapan model 1
        const nearest = await findNearestEarthquake(location.lat, location.lng);
        console.log("Gempa terdekat:", nearest as Earthquake);
        // Di sini kamu bisa menyimpan hasilnya ke state untuk ditampilkan di UI
        setNearestEarthquake(nearest as Earthquake);
        const nearestAddress = nearest
          ? await getAddress(nearest.latitude, nearest.longitude)
          : "";
        setNearestEarthquakeAddress(nearestAddress);
        const stats = await getAreaStatistics(location.lat, location.lng);
        setAreaStats(stats);
        // Mulai inferensi model 1
        const res: PredictionResponse | null = await fetchHazardPrediction(
          location.lat,
          location.lng,
        );
        setPredictionModel1(res);

        // Untuk penerapan model 2 (misalnya prediksi presentase kejadian gempa mendatang)
        const resModel2: PredictionM5RadiusResponse | null =
          await fetchM5RadiusPrediction(location.lat, location.lng, radiusKm);
        setPredictionModel2(resModel2);
      }
    },
    [radiusKm],
  );

  const mulaiAnalisis = useCallback(async () => {
    if (clickedLocation) {
      await runAnalisis(clickedLocation);
    } else {
      alert(
        "Silakan klik pada peta untuk memilih lokasi yang ingin dianalisis.",
      );
    }
  }, [clickedLocation, runAnalisis]);

  useEffect(() => {
    if (clickedLocation) {
      runAnalisis(clickedLocation);
    }
  }, [clickedLocation, runAnalisis]);

  // reset lokasi, melepas mark dan atur zoom
  const resetLocation = () => {
    setClickedLocation(null);
    setClickedAddress("");
    setNearestEarthquake(null);
    setNearestEarthquakeAddress("");
    setAreaStats(null);
    setPredictionModel1(null);
    setPredictionModel2(null);
  };

  const getM5ExpectedEventSummary = (value: number) => {
    if (value < 0.5) {
      return {
        approx: 0,
        text: "Nilai ini cenderung berarti dalam 1 bulan bisa tidak terjadi kejadian, tetapi kemungkinan 1 kejadian masih tetap ada.",
      };
    }

    if (value < 1.5) {
      return {
        approx: 1,
        text: "Nilai ini dapat dibaca sebagai perkiraan sekitar 1 kejadian gempa M>=5 dalam 1 bulan.",
      };
    }

    const approx = Math.round(value);
    return {
      approx,
      text: `Nilai ini dapat dibaca sebagai perkiraan sekitar ${approx} kejadian gempa M>=5 dalam 1 bulan.`,
    };
  };

  return (
    <div className="flex flex-col w-full overflow-hidden bg-gray-100">
      {/* Header */}
      {/* <header className=" text-white p-10 z-10">
        <h1 className="pl-20 text-xl font-bold tracking-tight"></h1>
      </header> */}

      {/* jumbotron */}
      <div className="text-white text-center relative">
        <img
          src="./images/earthquake-pictures.jpg"
          alt="..."
          className="absolute z-1 w-full h-full object-cover "
        />
        <div className="after absolute z-2 w-full h-full bg-linear-to-tr from-[rgba(0,0,0,0.9)] to-[rgba(0,0,0,0.2)]"></div>
        <div className="relative z-2 pt-14 lg:pt-32 px-4 pb-10 font-(Inter)">
          <h2 className="text-4xl font-bold mb-2 text-shadow-lg pb-3">
            Selamat Datang di{" "}
            <span className="text-amber-300">Peta Potensi Gempa Indonesia</span>
          </h2>
          <p className="text-base lg:text-2xl lg:px-60 xl:px-96 font-light text-shadow-lg pb-8 lg:pb-10">
            Pantau potensi bencana gempa di lingkungan sekitar kita dengan mudah
            dan cepat.
          </p>

          <div className="flex lg:px-20 gap-10 lg:gap-14 flex-col lg:flex-row justify-center items-center lg:items-start pb-10">
            <div className="flex-1 max-w-120">
              <p className="text-center mb-4 text-3xl">⚠️</p>
              <h3 className="text-base lg:text-xl font-bold mb-2 text-shadow-lg text-center lg:text-center">
                Potensi Tingkat Bahaya Dan Dampak Gempa
              </h3>
              <p className="text-sm font-light text-shadow-lg text-justify lg:text-justify">
                Kita dapat dengan mudah memprediksi potensi dampak gempa di
                lingkungan sekitar berdasarkan data kejadian gempa sebelumnya.
                Dengan memanfaatkan data historis dan model prediktif, kita
                dapat memberikan informasi yang berguna untuk meningkatkan
                kewaspadaan dan kesiapsiagaan terhadap potensi bencana gempa.
              </p>
            </div>
            <div className="flex-1 max-w-120">
              <p className="text-center mb-4 text-3xl">📈</p>
              <h3 className="text-base lg:text-xl font-bold mb-2 text-shadow-lg text-center lg:text-center">
                Prediksi Persentase Kejadian Gempa Mendatang
              </h3>
              <p className="text-sm font-light text-shadow-lg text-justify lg:text-justify">
                Model prediksi timeseries menganalisis pola historis gempa dari
                waktu ke waktu untuk meramalkan kemungkinan terjadinya gempa di
                masa depan. Dengan teknologi pembelajaran mesin yang canggih,
                sistem dapat mengidentifikasi tren temporal dan pola musiman
                untuk memberikan perkiraan tingkat kejadian gempa yang akurat
                pada periode mendatang.
              </p>
            </div>
          </div>
          {/* <div className="btn-double-effect bg-amber-200"></div> */}
          <button
            onClick={handleScrollToMain}
            className="relative my-10 px-4 py-2 text-lg bg-black font-bold rounded-3xl hover:bg-linear-to-r from-[#ef9917] to-[#e2eb60] cursor-pointer hover:text-gray-700 transition-colors
          after:content-['']
          after:absolute 
          after:top-1/2 
          after:left-1/2
          after:translate-x-[-50%]
          after:translate-y-[-50%]
          after:w-full
          after:h-full
          after:z-[-1]
          after:box-content
          after:p-0.5
          after:rounded-[inherit]
          after:bg-[conic-gradient(from_var(--angle),transparent_70%,#fafafa)]
          after:blur-[10px]
          after:opacity-[0.6]

          before:content-['']
          before:absolute 
          before:top-1/2 
          before:left-1/2
          before:translate-x-[-50%]
          before:translate-y-[-50%]
          before:w-full
          before:h-full
          before:z-[-1]
          before:box-content
          before:p-0.5
          before:rounded-[inherit]
          before:bg-[conic-gradient(from_var(--angle),transparent_70%,#fafafa)]
          
          after:animate-spin-slow
          before:animate-spin-slow
          "
          >
            Coba Sekarang
          </button>
        </div>
      </div>

      {/* Main Content */}

      <main
        id="main"
        className="lg:pt-10 flex flex-col-reverse lg:flex-row relative px-5 lg:px-10 xl:px-40 gap-5 pb-5 pt-5"
      >
        {/* Peta (Kiri) */}
        <div className="lg:grow lg:w-96 z-0 bg-slate-500 rounded-lg h-125 ">
          <MapWithNoSSR
            currentLocation={currentLocation}
            clickedLocation={clickedLocation}
            setClickedLocation={setClickedLocation}
          />
        </div>

        {/* Info Panel (Kanan) */}
        <aside className="flex-1 bg-white z-20 p-6 overflow-y-auto rounded-lg">
          <h1 className="text-lg mb-3 font-semibold">Petunjuk Penggunaan</h1>
          <ol className="pl-5 list-decimal list-outside" type="1">
            <li className="text-sm text-gray-600 mb-2">
              Klik pada peta di Indonesia untuk memilih lokasi yang ingin Anda
              analisis dengan mengeklik suatu wilayah di peta.{" "}
              <span className="text-orange-500">
                Jangan mengeklik peta di luar Indonesia, karena hasil
                informasinya tidak akan akurat
              </span>
            </li>
            <li className="text-sm text-gray-600 mb-2">
              Setelah memilih lokasi di peta, sistem akan otomatis memproses dua
              model prediksi. Tombol orange tetap tersedia jika Anda ingin
              refresh analisis secara manual pada titik yang sama.
            </li>
            <li className="text-sm text-gray-600 mb-2">
              Hasil analisis timeseries akan ditampilkan di bagian bawah
              halaman, termasuk grafik prediksi dengan perkiraan frekuensi gempa
              per bulan, interval kepercayaan model, evaluasi performa, dan
              visualisasi data historis yang digunakan.....
            </li>
            <li className="text-sm text-gray-600 mb-2">
              Gunakan informasi ini untuk meningkatkan kewaspadaan dan
              kesiapsiagaan terhadap potensi bencana gempa di lingkungan sekitar
              Anda.
            </li>
          </ol>
        </aside>
      </main>
      {/* aksi tombol */}
      <div className="px-5 lg:px-40 mb-10">
        <div className="mb-4">
          <div className="flex">
            <h1 className="">📍</h1>
            <h1 className="font-semibold text-black ">
              Lokasi saat ini:
              <span className="font-normal text-sm lg:text-base text-gray-500">
                {" "}
                {currentAddress}.
              </span>
            </h1>
          </div>
          <div className="flex">
            <h1 className="">📍</h1>
            <h1 className="font-semibold text-black ">
              Lokasi yang dipilih:
              <span className="font-normal text-sm lg:text-base text-gray-500">
                {" "}
                {clickedAddress == ""
                  ? "Belum ada yang diklik."
                  : clickedAddress}
                .
              </span>
            </h1>
          </div>
        </div>
        <div className="flex justify-center gap-5">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-700" htmlFor="radius-km">
              Radius (50-200 km)
            </label>
            <input
              id="radius-km"
              type="number"
              min={50}
              max={200}
              step={10}
              value={radiusKm}
              onChange={(e) => {
                const nextValue = Number(e.target.value);
                if (!Number.isNaN(nextValue)) {
                  setRadiusKm(Math.max(50, Math.min(200, nextValue)));
                }
              }}
              className="w-24 rounded-md border border-gray-300 px-2 py-1 text-sm"
            />
          </div>
          <button
            className="isolate relative text-md rounded-2xl font-semibold
          "
            onClick={mulaiAnalisis}
          >
            <div className="bg-amber-400 z-10 py-2 px-4 rounded-[inherit]">
              Mulai Analisis Lokasi
            </div>
          </button>
          <button
            className="border border-black text-md rounded-2xl py-2 px-4"
            onClick={resetLocation}
          >
            Reset Lokasi
          </button>
        </div>
      </div>
      {/* hasil */}
      <div className="px-5 lg:px-10 xl:px-40 mb-20">
        <h2 className="text-3xl font-bold mb-4 text-center">
          Hasil Analisis Lokasi
        </h2>
        <div className="flex flex-col lg:flex-row justify-center items-center lg:items-start gap-5">
          <div className="flex-1 p-6 bg-white rounded-lg ">
            <h3 className="text-xl font-semibold mb-2">
              📊 Potensi Dampak Gempa Dari Kejadian Lampau
            </h3>
            {predictionModel1 && (
              <>
                <p className="text-sm mb-4">
                  Berdasarkan data historis dan model prediktif, berikut adalah
                  potensi dampak bencana untuk lokasi yang Anda pilih:
                </p>

                {/* low medium high very_high */}
                <p className="text-md mb-4">
                  <b>Hasil:</b> Lokasi tergolong area dengan potensial dampak
                  gempa{" "}
                  <span
                    className={
                      String(predictionModel1.data.hazard_level) == "low"
                        ? "text-green-600 font-bold"
                        : String(predictionModel1.data.hazard_level) == "medium"
                          ? "text-yellow-600 font-bold"
                          : String(predictionModel1.data.hazard_level) == "high"
                            ? "text-orange-600 font-bold"
                            : String(predictionModel1.data.hazard_level) ==
                                "very_high"
                              ? "text-red-700 font-bold"
                              : ""
                    }
                  >
                    {String(predictionModel1.data.hazard_level) == "low"
                      ? "Rendah"
                      : String(predictionModel1.data.hazard_level) == "medium"
                        ? "Sedang"
                        : String(predictionModel1.data.hazard_level) == "high"
                          ? "Tinggi"
                          : String(predictionModel1.data.hazard_level) ==
                              "very_high"
                            ? "Sangat Tinggi"
                            : ""}
                  </span>
                </p>
              </>
            )}

            {nearestEarthquake && (
              <p className="text-sm mb-1">
                Lokasi cukup dekat dengan zona gempa {nearestEquakeAddress},
                yang terjadi pada{" "}
                {formatTanggalIndo(nearestEarthquake?.datetime)} dengan
                magnitudo sebesar {nearestEarthquake?.magnitude.toFixed(2)}{" "}
                Richter. Dengan hasil analisis data sebagai berikut:
              </p>
            )}

            {areaStats && (
              <ul className="list-disc list-inside text-gray-600 text-sm">
                <li>
                  Frekuensi terjadinya gempa di area sekitar:{" "}
                  {areaStats?.frequency}
                  {" kali"}
                </li>
                <li>
                  Magnitudo terbesar yang pernah terjadi:{" "}
                  {areaStats?.maxMagnitude} Richter
                </li>
                <li>Rata-rata magnitudo: {areaStats?.avgMagnitude}</li>
                <li>
                  Rata-rata kedalaman sumber gempa: {areaStats?.avgDepth}
                  {" km"}
                </li>
              </ul>
            )}

            <p className="text-xs text-gray-500 mt-4">
              *Tingkatan prediksi dampak gempa dihitung berdasarkan kombinasi
              data historis dan model prediktif <i>Machine Learning.</i>
            </p>
          </div>
          <div className="flex-1 p-6 bg-white rounded-lg">
            <h3 className="text-xl font-semibold mb-2">
              📈 Potensi Kejadian Gempa Magnitude&gt;=5 (Akumulasi 1 Bulan)
            </h3>
            {!predictionModel2 && (
              <p className="text-sm text-gray-600">
                Atur radius (50-200 km), lalu klik titik lokasi di peta untuk
                melihat potensi kejadian gempa M&gt;=5 di sekitar lokasi.
              </p>
            )}

            {predictionModel2 && (
              <>
                <p className="text-sm mb-2">
                  Berdasarkan data historis dan model prediktif, dalam radius{" "}
                  <b>{predictionModel2.data.radius_km} km </b> dari titik yang
                  Anda pilih, estimasi akumulasi kejadian gempa M&gt;=5 untuk 1
                  bulan ke depan adalah{" "}
                  <b>
                    {predictionModel2.data.estimated_total_m5_in_radius.toFixed(
                      2,
                    )}
                  </b>{" "}
                  kejadian.
                </p>

                <p className="text-sm mb-2">
                  <b>Interpretasi:</b> Angka di atas adalah nilai ekspektasi
                  (rata-rata prediksi), bukan jumlah kejadian yang pasti.
                  Semakin besar angkanya, semakin besar potensi dampak kejadian
                  gempa M&gt;=5 pada bulan target.
                </p>

                <p className="text-sm mb-3">
                  {(() => {
                    const summary = getM5ExpectedEventSummary(
                      predictionModel2.data.estimated_total_m5_in_radius,
                    );
                    return (
                      <>
                        <b>Perkiraan sederhana:</b> Nilai{" "}
                        <b>
                          {predictionModel2.data.estimated_total_m5_in_radius.toFixed(
                            2,
                          )}
                        </b>{" "}
                        mendekati <b>{summary.approx}</b> kejadian.{" "}
                        {summary.text}
                      </>
                    );
                  })()}
                </p>

                <ul className="list-disc list-inside text-gray-600 text-sm">
                  <li>
                    Wilayah acuan terdekat model:{" "}
                    {predictionModel2.data.nearest_region}
                  </li>
                  <li>
                    Jarak ke pusat wilayah acuan:{" "}
                    {predictionModel2.data.distance_km.toFixed(1)} km
                  </li>
                  <li>
                    Jumlah wilayah yang terhitung dalam radius:{" "}
                    {predictionModel2.data.n_regions_in_radius}
                  </li>
                  <li>
                    Bulan analisis dilakukan: {predictionModel2.data.open_month}
                  </li>
                  <li>
                    Periode prediksi: {predictionModel2.data.target_month}
                  </li>
                  <li>
                    Data observasi terakhir model:{" "}
                    {predictionModel2.data.model_last_observed_month}
                  </li>
                </ul>

                <p className="text-xs text-gray-500 mt-4">
                  *Prediksi ini membantu membaca potensi dampak secara dini,
                  bukan sebagai kepastian waktu terjadinya gempa.
                </p>
              </>
            )}
          </div>
        </div>
      </div>
      <div className="text-center text-sm text-gray-500 mb-5">
        <h1>
          copyright &copy; 2026 Peta Potensi Gempa Indonesia. All rights
          reserved.
        </h1>
      </div>
    </div>
  );
}
