import { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Definisi Tipe Data
interface Location {
  lat: number;
  lng: number;
}

interface MapProps {
  currentLocation: Location | null;
  clickedLocation: Location | null;
  setClickedLocation: (loc: Location) => void;
}

// Fix Icon Leaflet di Next.js
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Komponen Handler Klik
function ClickHandler({ setClickedLocation }: { setClickedLocation: (loc: Location) => void }) {
  useMapEvents({
    click(e) {
      setClickedLocation({ lat: e.latlng.lat, lng: e.latlng.lng });
    },
  });
  return null;
}

// Komponen Auto-Recenter Kamera
function RecenterAutomatically({ lat, lng }: Location) {
  const map = useMap();
  useEffect(() => {
    if (lat && lng) {
      map.setView([lat, lng], 13);
    }
  }, [lat, lng, map]);
  return null;
}

export default function MapComponent({ currentLocation, clickedLocation, setClickedLocation }: MapProps) {
  const defaultCenter: [number, number] = [-6.2000, 106.8166]; // Jakarta

  return (
    <MapContainer center={defaultCenter} zoom={5} className='w-full h-full rounded-lg border-2 border-slate-600' >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      
      {currentLocation && (
        <>
          <Marker position={[currentLocation.lat, currentLocation.lng]}>
            <Popup>Lokasi Anda Sekarang</Popup>
          </Marker>
          <RecenterAutomatically lat={currentLocation.lat} lng={currentLocation.lng} />
        </>
      )}

      {clickedLocation && (
        <Marker position={[clickedLocation.lat, clickedLocation.lng]}>
          <Popup>Titik yang dipilih</Popup>
        </Marker>
      )}

      <ClickHandler setClickedLocation={setClickedLocation} />
    </MapContainer>
  );
}