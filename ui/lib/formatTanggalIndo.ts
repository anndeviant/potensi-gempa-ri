const formatTanggalIndo = (dateString: string) => {
  const date = new Date(dateString);

  // Format Tanggal: 12 Juni 2016
  const opsiTanggal: Intl.DateTimeFormatOptions = {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  };

  // Format Jam: 01:32
  const opsiJam: Intl.DateTimeFormatOptions = {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: 'Asia/Jakarta' // Memastikan output adalah WIB
  };

  const tanggal = new Intl.DateTimeFormat('id-ID', opsiTanggal).format(date);
  const jam = new Intl.DateTimeFormat('id-ID', opsiJam).format(date);

  return `${tanggal} jam ${jam} WIB`;
};

export default formatTanggalIndo;