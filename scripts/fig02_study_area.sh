#!/usr/bin/env bash
set -euo pipefail

SHP=${1:-.}
REG=101.9/111.7/8/23.6
RDEM=101.9/111.8/8/23.6
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

gmt grdcut @earth_relief_03s -R$RDEM -G"$WORK/dem03.nc"
gmt grdgradient "$WORK/dem03.nc" -A315 -Ne0.6 -G"$WORK/int03.nc"
for v in dem int; do
  gmt grdfilter "$WORK/${v}03.nc" -Fb0.4 -D1 -G"$WORK/f_$v.nc"
  gmt grdsample "$WORK/f_$v.nc" -I12s -R$RDEM -G"$WORK/$v.nc"
done
gmt grdclip "$WORK/dem.nc" -Sb0/0 -G"$WORK/land.nc"
gmt makecpt -Cgeo -T0/3000/100 -Z > "$WORK/relief.cpt"

ogr2ogr -f OGR_GMT -simplify 0.001 "$WORK/vn0.gmt" "$SHP/vnm_admin0.shp"
ogr2ogr -f OGR_GMT -simplify 0.001 "$WORK/vn1.gmt" "$SHP/vnm_admin1.shp"

L=$(python3 -c "import math; print(round(200/(111.32*math.cos(math.radians(9.6))),4))")
X0=109.35
X1=$(python3 -c "print($X0+$L)")
XM=$(python3 -c "print($X0+$L/2)")

H="black=~1.2p,white"
gmt begin fig02_study_area pdf,png
  gmt set FONT_ANNOT_PRIMARY 8p,Helvetica FONT_LABEL 9p,Helvetica MAP_FRAME_TYPE fancy MAP_FRAME_WIDTH 2.5p \
          MAP_GRID_PEN_PRIMARY 0.25p,white MAP_TICK_LENGTH_PRIMARY 3p FORMAT_GEO_MAP ddd:mmF \
          PS_CONVERT="A+m0.1c,E600"
  gmt basemap -R$REG -JM9c -Bxa2f1g2 -Bya2f1g2 -BWSne
  gmt coast -S221/233/242 -Di -A50
  gmt coast -Gc -Di -A50
  gmt grdimage "$WORK/land.nc" -C"$WORK/relief.cpt" -I"$WORK/int.nc" -t55
  gmt coast -Q
  gmt clip "$WORK/vn0.gmt"
  gmt grdimage "$WORK/land.nc" -C"$WORK/relief.cpt" -I"$WORK/int.nc"
  gmt clip -C
  gmt coast -W0.2p,80/100/120 -Di -A50
  gmt plot "$WORK/vn1.gmt" -W0.25p,50
  gmt plot "$WORK/vn0.gmt" -W0.7p,black
  gmt basemap -Bxg2 -Byg2

  gmt text -F+f+j+a <<EOF
103.9 22.95 8p,Helvetica-Oblique,70 CM 0 CHINA
103.6 18.9 8p,Helvetica-Oblique,70 CM 0 LAOS
102.7 16.0 8p,Helvetica-Oblique,70 CM 0 THAILAND
104.6 12.6 8p,Helvetica-Oblique,70 CM 0 CAMBODIA
107.45 20.2 7.5p,Helvetica-Oblique,40/80/120 CM 0 Gulf of
107.45 19.9 7.5p,Helvetica-Oblique,40/80/120 CM 0 Tonkin
103.3 9.6 7.5p,Helvetica-Oblique,40/80/120 CM 90 Gulf of Thailand
110.3 12.2 7.5p,Helvetica-Oblique,40/80/120 CM 0 East Sea
110.3 11.85 6.5p,Helvetica-Oblique,40/80/120 CM 0 (South China Sea)
105.25 21.20 7.5p,Helvetica-Bold,$H RM 0 Hanoi
107.00 10.55 7.5p,Helvetica-Bold,$H LM 0 Ho Chi Minh City
108.42 16.10 7.5p,Helvetica-Bold,$H LM 0 Da Nang
106.55 20.62 7p,Helvetica-Oblique,$H CM 0 Red River
106.55 20.35 7p,Helvetica-Oblique,$H CM 0 Delta
105.55 9.75 7p,Helvetica-Oblique,$H CM 0 Mekong Delta
108.15 13.60 7p,Helvetica-Oblique,$H CM 0 Central
108.15 13.33 7p,Helvetica-Oblique,$H CM 0 Highlands
105.2 18.45 7p,Helvetica-Oblique,$H CM -47 Truong Son Range
104.2 22.0 7p,Helvetica-Oblique,$H CM 0 Northwest
104.2 21.73 7p,Helvetica-Oblique,$H CM 0 Uplands
EOF
  printf "105.85 21.03\n106.70 10.78\n108.22 16.05\n" | gmt plot -Sc0.13c -Gblack -W0.4p,white

  gmt colorbar -C"$WORK/relief.cpt" -DjBR+w3.6c/0.22c+o0.35c/0.55c+h+ml -Bxa1000f250 -Bx+l"Elevation (m)" \
      --FONT_ANNOT_PRIMARY=7p,Helvetica --FONT_LABEL=7p,Helvetica

  printf "$X0 9.6\n$XM 9.6\n" | gmt plot -W3p,black
  printf "$XM 9.6\n$X1 9.6\n" | gmt plot -W3p,black
  printf "$XM 9.6\n$X1 9.6\n" | gmt plot -W1.8p,white
  printf ">\n$X0 9.52\n$X0 9.68\n>\n$X1 9.52\n$X1 9.68\n" | gmt plot -W0.5p,black
  printf "$X0 9.78 0\n$X1 9.78 200 km\n" | gmt text -F+f7p,Helvetica,black+jCB

  gmt basemap -TdjTR+w0.9c+f1+l,,,N+o0.35c/0.2c

  gmt inset begin -DjTR+w2.4c+o0.25c/1.7c
    gmt coast -Rg -JG106/15/2.4c -G210 -S235/242/248 -A5000 -Bg30 --MAP_GRID_PEN_PRIMARY=0.2p,gray60
    gmt coast -EVN+gred3 -A5000
  gmt inset end
gmt end
echo "written: fig02_study_area.pdf, fig02_study_area.png"
