    const zoneInfo = [{"id":1,"name":"Newark Airport","borough":"EWR"},{"id":2,"name":"Jamaica Bay","borough":"Queens"},{"id":3,"name":"Allerton/Pelham Gardens","borough":"Bronx"},{"id":4,"name":"Alphabet City","borough":"Manhattan"},{"id":5,"name":"Arden Heights","borough":"Staten Island"},{"id":6,"name":"Arrochar/Fort Wadsworth","borough":"Staten Island"},{"id":7,"name":"Astoria","borough":"Queens"},{"id":8,"name":"Astoria Park","borough":"Queens"},{"id":9,"name":"Auburndale","borough":"Queens"},{"id":10,"name":"Baisley Park","borough":"Queens"},{"id":11,"name":"Bath Beach","borough":"Brooklyn"},{"id":12,"name":"Battery Park","borough":"Manhattan"},{"id":13,"name":"Battery Park City","borough":"Manhattan"},{"id":14,"name":"Bay Ridge","borough":"Brooklyn"},{"id":15,"name":"Bay Terrace/Fort Totten","borough":"Queens"},{"id":16,"name":"Bayside","borough":"Queens"},{"id":17,"name":"Bedford","borough":"Brooklyn"},{"id":18,"name":"Bedford Park","borough":"Bronx"},{"id":19,"name":"Bellerose","borough":"Queens"},{"id":20,"name":"Belmont","borough":"Bronx"},{"id":21,"name":"Bensonhurst East","borough":"Brooklyn"},{"id":22,"name":"Bensonhurst West","borough":"Brooklyn"},{"id":23,"name":"Bloomfield/Emerson Hill","borough":"Staten Island"},{"id":24,"name":"Bloomingdale","borough":"Manhattan"},{"id":25,"name":"Boerum Hill","borough":"Brooklyn"},{"id":26,"name":"Borough Park","borough":"Brooklyn"},{"id":27,"name":"Breezy Point/Fort Tilden/Riis Beach","borough":"Queens"},{"id":28,"name":"Briarwood/Jamaica Hills","borough":"Queens"},{"id":29,"name":"Brighton Beach","borough":"Brooklyn"},{"id":30,"name":"Broad Channel","borough":"Queens"},{"id":31,"name":"Bronx Park","borough":"Bronx"},{"id":32,"name":"Bronxdale","borough":"Bronx"},{"id":33,"name":"Brooklyn Heights","borough":"Brooklyn"},{"id":34,"name":"Brooklyn Navy Yard","borough":"Brooklyn"},{"id":35,"name":"Brownsville","borough":"Brooklyn"},{"id":36,"name":"Bushwick North","borough":"Brooklyn"},{"id":37,"name":"Bushwick South","borough":"Brooklyn"},{"id":38,"name":"Cambria Heights","borough":"Queens"},{"id":39,"name":"Canarsie","borough":"Brooklyn"},{"id":40,"name":"Carroll Gardens","borough":"Brooklyn"},{"id":41,"name":"Central Harlem","borough":"Manhattan"},{"id":42,"name":"Central Harlem North","borough":"Manhattan"},{"id":43,"name":"Central Park","borough":"Manhattan"},{"id":44,"name":"Charleston/Tottenville","borough":"Staten Island"},{"id":45,"name":"Chinatown","borough":"Manhattan"},{"id":46,"name":"City Island","borough":"Bronx"},{"id":47,"name":"Claremont/Bathgate","borough":"Bronx"},{"id":48,"name":"Clinton East","borough":"Manhattan"},{"id":49,"name":"Clinton Hill","borough":"Brooklyn"},{"id":50,"name":"Clinton West","borough":"Manhattan"},{"id":51,"name":"Co-Op City","borough":"Bronx"},{"id":52,"name":"Cobble Hill","borough":"Brooklyn"},{"id":53,"name":"College Point","borough":"Queens"},{"id":54,"name":"Columbia Street","borough":"Brooklyn"},{"id":55,"name":"Coney Island","borough":"Brooklyn"},{"id":56,"name":"Corona","borough":"Queens"},{"id":57,"name":"Corona","borough":"Queens"},{"id":58,"name":"Country Club","borough":"Bronx"},{"id":59,"name":"Crotona Park","borough":"Bronx"},{"id":60,"name":"Crotona Park East","borough":"Bronx"},{"id":61,"name":"Crown Heights North","borough":"Brooklyn"},{"id":62,"name":"Crown Heights South","borough":"Brooklyn"},{"id":63,"name":"Cypress Hills","borough":"Brooklyn"},{"id":64,"name":"Douglaston","borough":"Queens"},{"id":65,"name":"Downtown Brooklyn/MetroTech","borough":"Brooklyn"},{"id":66,"name":"DUMBO/Vinegar Hill","borough":"Brooklyn"},{"id":67,"name":"Dyker Heights","borough":"Brooklyn"},{"id":68,"name":"East Chelsea","borough":"Manhattan"},{"id":69,"name":"East Concourse/Concourse Village","borough":"Bronx"},{"id":70,"name":"East Elmhurst","borough":"Queens"},{"id":71,"name":"East Flatbush/Farragut","borough":"Brooklyn"},{"id":72,"name":"East Flatbush/Remsen Village","borough":"Brooklyn"},{"id":73,"name":"East Flushing","borough":"Queens"},{"id":74,"name":"East Harlem North","borough":"Manhattan"},{"id":75,"name":"East Harlem South","borough":"Manhattan"},{"id":76,"name":"East New York","borough":"Brooklyn"},{"id":77,"name":"East New York/Pennsylvania Avenue","borough":"Brooklyn"},{"id":78,"name":"East Tremont","borough":"Bronx"},{"id":79,"name":"East Village","borough":"Manhattan"},{"id":80,"name":"East Williamsburg","borough":"Brooklyn"},{"id":81,"name":"Eastchester","borough":"Bronx"},{"id":82,"name":"Elmhurst","borough":"Queens"},{"id":83,"name":"Elmhurst/Maspeth","borough":"Queens"},{"id":84,"name":"Eltingville/Annadale/Prince's Bay","borough":"Staten Island"},{"id":85,"name":"Erasmus","borough":"Brooklyn"},{"id":86,"name":"Far Rockaway","borough":"Queens"},{"id":87,"name":"Financial District North","borough":"Manhattan"},{"id":88,"name":"Financial District South","borough":"Manhattan"},{"id":89,"name":"Flatbush/Ditmas Park","borough":"Brooklyn"},{"id":90,"name":"Flatiron","borough":"Manhattan"},{"id":91,"name":"Flatlands","borough":"Brooklyn"},{"id":92,"name":"Flushing","borough":"Queens"},{"id":93,"name":"Flushing Meadows-Corona Park","borough":"Queens"},{"id":94,"name":"Fordham South","borough":"Bronx"},{"id":95,"name":"Forest Hills","borough":"Queens"},{"id":96,"name":"Forest Park/Highland Park","borough":"Queens"},{"id":97,"name":"Fort Greene","borough":"Brooklyn"},{"id":98,"name":"Fresh Meadows","borough":"Queens"},{"id":99,"name":"Freshkills Park","borough":"Staten Island"},{"id":100,"name":"Garment District","borough":"Manhattan"},{"id":101,"name":"Glen Oaks","borough":"Queens"},{"id":102,"name":"Glendale","borough":"Queens"},{"id":103,"name":"Governor's Island/Ellis Island/Liberty Island","borough":"Manhattan"},{"id":104,"name":"Governor's Island/Ellis Island/Liberty Island","borough":"Manhattan"},{"id":105,"name":"Governor's Island/Ellis Island/Liberty Island","borough":"Manhattan"},{"id":106,"name":"Gowanus","borough":"Brooklyn"},{"id":107,"name":"Gramercy","borough":"Manhattan"},{"id":108,"name":"Gravesend","borough":"Brooklyn"},{"id":109,"name":"Great Kills","borough":"Staten Island"},{"id":110,"name":"Great Kills Park","borough":"Staten Island"},{"id":111,"name":"Green-Wood Cemetery","borough":"Brooklyn"},{"id":112,"name":"Greenpoint","borough":"Brooklyn"},{"id":113,"name":"Greenwich Village North","borough":"Manhattan"},{"id":114,"name":"Greenwich Village South","borough":"Manhattan"},{"id":115,"name":"Grymes Hill/Clifton","borough":"Staten Island"},{"id":116,"name":"Hamilton Heights","borough":"Manhattan"},{"id":117,"name":"Hammels/Arverne","borough":"Queens"},{"id":118,"name":"Heartland Village/Todt Hill","borough":"Staten Island"},{"id":119,"name":"Highbridge","borough":"Bronx"},{"id":120,"name":"Highbridge Park","borough":"Manhattan"},{"id":121,"name":"Hillcrest/Pomonok","borough":"Queens"},{"id":122,"name":"Hollis","borough":"Queens"},{"id":123,"name":"Homecrest","borough":"Brooklyn"},{"id":124,"name":"Howard Beach","borough":"Queens"},{"id":125,"name":"Hudson Sq","borough":"Manhattan"},{"id":126,"name":"Hunts Point","borough":"Bronx"},{"id":127,"name":"Inwood","borough":"Manhattan"},{"id":128,"name":"Inwood Hill Park","borough":"Manhattan"},{"id":129,"name":"Jackson Heights","borough":"Queens"},{"id":130,"name":"Jamaica","borough":"Queens"},{"id":131,"name":"Jamaica Estates","borough":"Queens"},{"id":132,"name":"JFK Airport","borough":"Queens"},{"id":133,"name":"Kensington","borough":"Brooklyn"},{"id":134,"name":"Kew Gardens","borough":"Queens"},{"id":135,"name":"Kew Gardens Hills","borough":"Queens"},{"id":136,"name":"Kingsbridge Heights","borough":"Bronx"},{"id":137,"name":"Kips Bay","borough":"Manhattan"},{"id":138,"name":"LaGuardia Airport","borough":"Queens"},{"id":139,"name":"Laurelton","borough":"Queens"},{"id":140,"name":"Lenox Hill East","borough":"Manhattan"},{"id":141,"name":"Lenox Hill West","borough":"Manhattan"},{"id":142,"name":"Lincoln Square East","borough":"Manhattan"},{"id":143,"name":"Lincoln Square West","borough":"Manhattan"},{"id":144,"name":"Little Italy/NoLiTa","borough":"Manhattan"},{"id":145,"name":"Long Island City/Hunters Point","borough":"Queens"},{"id":146,"name":"Long Island City/Queens Plaza","borough":"Queens"},{"id":147,"name":"Longwood","borough":"Bronx"},{"id":148,"name":"Lower East Side","borough":"Manhattan"},{"id":149,"name":"Madison","borough":"Brooklyn"},{"id":150,"name":"Manhattan Beach","borough":"Brooklyn"},{"id":151,"name":"Manhattan Valley","borough":"Manhattan"},{"id":152,"name":"Manhattanville","borough":"Manhattan"},{"id":153,"name":"Marble Hill","borough":"Manhattan"},{"id":154,"name":"Marine Park/Floyd Bennett Field","borough":"Brooklyn"},{"id":155,"name":"Marine Park/Mill Basin","borough":"Brooklyn"},{"id":156,"name":"Mariners Harbor","borough":"Staten Island"},{"id":157,"name":"Maspeth","borough":"Queens"},{"id":158,"name":"Meatpacking/West Village West","borough":"Manhattan"},{"id":159,"name":"Melrose South","borough":"Bronx"},{"id":160,"name":"Middle Village","borough":"Queens"},{"id":161,"name":"Midtown Center","borough":"Manhattan"},{"id":162,"name":"Midtown East","borough":"Manhattan"},{"id":163,"name":"Midtown North","borough":"Manhattan"},{"id":164,"name":"Midtown South","borough":"Manhattan"},{"id":165,"name":"Midwood","borough":"Brooklyn"},{"id":166,"name":"Morningside Heights","borough":"Manhattan"},{"id":167,"name":"Morrisania/Melrose","borough":"Bronx"},{"id":168,"name":"Mott Haven/Port Morris","borough":"Bronx"},{"id":169,"name":"Mount Hope","borough":"Bronx"},{"id":170,"name":"Murray Hill","borough":"Manhattan"},{"id":171,"name":"Murray Hill-Queens","borough":"Queens"},{"id":172,"name":"New Dorp/Midland Beach","borough":"Staten Island"},{"id":173,"name":"North Corona","borough":"Queens"},{"id":174,"name":"Norwood","borough":"Bronx"},{"id":175,"name":"Oakland Gardens","borough":"Queens"},{"id":176,"name":"Oakwood","borough":"Staten Island"},{"id":177,"name":"Ocean Hill","borough":"Brooklyn"},{"id":178,"name":"Ocean Parkway South","borough":"Brooklyn"},{"id":179,"name":"Old Astoria","borough":"Queens"},{"id":180,"name":"Ozone Park","borough":"Queens"},{"id":181,"name":"Park Slope","borough":"Brooklyn"},{"id":182,"name":"Parkchester","borough":"Bronx"},{"id":183,"name":"Pelham Bay","borough":"Bronx"},{"id":184,"name":"Pelham Bay Park","borough":"Bronx"},{"id":185,"name":"Pelham Parkway","borough":"Bronx"},{"id":186,"name":"Penn Station/Madison Sq West","borough":"Manhattan"},{"id":187,"name":"Port Richmond","borough":"Staten Island"},{"id":188,"name":"Prospect-Lefferts Gardens","borough":"Brooklyn"},{"id":189,"name":"Prospect Heights","borough":"Brooklyn"},{"id":190,"name":"Prospect Park","borough":"Brooklyn"},{"id":191,"name":"Queens Village","borough":"Queens"},{"id":192,"name":"Queensboro Hill","borough":"Queens"},{"id":193,"name":"Queensbridge/Ravenswood","borough":"Queens"},{"id":194,"name":"Randalls Island","borough":"Manhattan"},{"id":195,"name":"Red Hook","borough":"Brooklyn"},{"id":196,"name":"Rego Park","borough":"Queens"},{"id":197,"name":"Richmond Hill","borough":"Queens"},{"id":198,"name":"Ridgewood","borough":"Queens"},{"id":199,"name":"Rikers Island","borough":"Bronx"},{"id":200,"name":"Riverdale/North Riverdale/Fieldston","borough":"Bronx"},{"id":201,"name":"Rockaway Park","borough":"Queens"},{"id":202,"name":"Roosevelt Island","borough":"Manhattan"},{"id":203,"name":"Rosedale","borough":"Queens"},{"id":204,"name":"Rossville/Woodrow","borough":"Staten Island"},{"id":205,"name":"Saint Albans","borough":"Queens"},{"id":206,"name":"Saint George/New Brighton","borough":"Staten Island"},{"id":207,"name":"Saint Michaels Cemetery/Woodside","borough":"Queens"},{"id":208,"name":"Schuylerville/Edgewater Park","borough":"Bronx"},{"id":209,"name":"Seaport","borough":"Manhattan"},{"id":210,"name":"Sheepshead Bay","borough":"Brooklyn"},{"id":211,"name":"SoHo","borough":"Manhattan"},{"id":212,"name":"Soundview/Bruckner","borough":"Bronx"},{"id":213,"name":"Soundview/Castle Hill","borough":"Bronx"},{"id":214,"name":"South Beach/Dongan Hills","borough":"Staten Island"},{"id":215,"name":"South Jamaica","borough":"Queens"},{"id":216,"name":"South Ozone Park","borough":"Queens"},{"id":217,"name":"South Williamsburg","borough":"Brooklyn"},{"id":218,"name":"Springfield Gardens North","borough":"Queens"},{"id":219,"name":"Springfield Gardens South","borough":"Queens"},{"id":220,"name":"Spuyten Duyvil/Kingsbridge","borough":"Bronx"},{"id":221,"name":"Stapleton","borough":"Staten Island"},{"id":222,"name":"Starrett City","borough":"Brooklyn"},{"id":223,"name":"Steinway","borough":"Queens"},{"id":224,"name":"Stuy Town/Peter Cooper Village","borough":"Manhattan"},{"id":225,"name":"Stuyvesant Heights","borough":"Brooklyn"},{"id":226,"name":"Sunnyside","borough":"Queens"},{"id":227,"name":"Sunset Park East","borough":"Brooklyn"},{"id":228,"name":"Sunset Park West","borough":"Brooklyn"},{"id":229,"name":"Sutton Place/Turtle Bay North","borough":"Manhattan"},{"id":230,"name":"Times Sq/Theatre District","borough":"Manhattan"},{"id":231,"name":"TriBeCa/Civic Center","borough":"Manhattan"},{"id":232,"name":"Two Bridges/Seward Park","borough":"Manhattan"},{"id":233,"name":"UN/Turtle Bay South","borough":"Manhattan"},{"id":234,"name":"Union Sq","borough":"Manhattan"},{"id":235,"name":"University Heights/Morris Heights","borough":"Bronx"},{"id":236,"name":"Upper East Side North","borough":"Manhattan"},{"id":237,"name":"Upper East Side South","borough":"Manhattan"},{"id":238,"name":"Upper West Side North","borough":"Manhattan"},{"id":239,"name":"Upper West Side South","borough":"Manhattan"},{"id":240,"name":"Van Cortlandt Park","borough":"Bronx"},{"id":241,"name":"Van Cortlandt Village","borough":"Bronx"},{"id":242,"name":"Van Nest/Morris Park","borough":"Bronx"},{"id":243,"name":"Washington Heights North","borough":"Manhattan"},{"id":244,"name":"Washington Heights South","borough":"Manhattan"},{"id":245,"name":"West Brighton","borough":"Staten Island"},{"id":246,"name":"West Chelsea/Hudson Yards","borough":"Manhattan"},{"id":247,"name":"West Concourse","borough":"Bronx"},{"id":248,"name":"West Farms/Bronx River","borough":"Bronx"},{"id":249,"name":"West Village","borough":"Manhattan"},{"id":250,"name":"Westchester Village/Unionport","borough":"Bronx"},{"id":251,"name":"Westerleigh","borough":"Staten Island"},{"id":252,"name":"Whitestone","borough":"Queens"},{"id":253,"name":"Willets Point","borough":"Queens"},{"id":254,"name":"Williamsbridge/Olinville","borough":"Bronx"},{"id":255,"name":"Williamsburg (North Side)","borough":"Brooklyn"},{"id":256,"name":"Williamsburg (South Side)","borough":"Brooklyn"},{"id":257,"name":"Windsor Terrace","borough":"Brooklyn"},{"id":258,"name":"Woodhaven","borough":"Queens"},{"id":259,"name":"Woodlawn/Wakefield","borough":"Bronx"},{"id":260,"name":"Woodside","borough":"Queens"},{"id":261,"name":"World Trade Center","borough":"Manhattan"},{"id":262,"name":"Yorkville East","borough":"Manhattan"},{"id":263,"name":"Yorkville West","borough":"Manhattan"}];
    async function startDashboard() {
    const feedMode = new URLSearchParams(location.search).get('feed') === 'replay' ? 'replay' : 'archive';
    const feedQuery = `?feed=${feedMode}`;
    const feedList = await fetch('/api/feeds', {cache:'no-store'}).then(response => response.json());
    document.getElementById('replay-link').hidden = !feedList.feeds.includes('replay');
    document.getElementById(`${feedMode}-link`).setAttribute('aria-current','page');
    document.getElementById('feed-badge').textContent = feedMode === 'replay' ? 'Advancing replay' : 'History archive';
    const initialResponse = await fetch(`/api/replay${feedQuery}`, {cache:'no-store'});
    if (!initialResponse.ok) {
      const problem = await initialResponse.json().catch(() => ({}));
      throw new Error(problem.error || `Data feed returned HTTP ${initialResponse.status}`);
    }
    const initialFeed = await initialResponse.json();
    if (initialFeed.zone_ids.length !== zoneInfo.length || !initialFeed.zone_ids.every((id,i) => id === zoneInfo[i].id)) {
      throw new Error('Data feed zone IDs do not match the map geometry');
    }
    let hours = initialFeed.hours;
    let feedVersion = initialFeed.version;
    const byId = new Map(zoneInfo.map((z,i) => [z.id,i]));
    const paths = [...document.querySelectorAll('.zone')];
    const demandColors = ['#1b3347','#255068','#2c7e93','#46a5a4','#b5b969','#ef9b51','#f2684d'];
    const gapColors = ['#253e50','#306177','#3d8895','#70a99b','#c8b66d','#e99652','#ed6250'];
    const wapeColors = ['#2b756d','#559c74','#a6b971','#dbb868','#e79a58','#dc7354','#cb4f55'];
    const noDataColor = '#26313d';
    const minWapeVolume = 50;
    const demandBucket = v => v < .5 ? 0 : v < 5 ? 1 : v < 15 ? 2 : v < 40 ? 3 : v < 100 ? 4 : v < 200 ? 5 : 6;
    const gapBucket = v => v < .5 ? 0 : v < 5 ? 1 : v < 15 ? 2 : v < 40 ? 3 : v < 80 ? 4 : v < 150 ? 5 : 6;
    const wapeBucket = v => v < 10 ? 0 : v < 20 ? 1 : v < 35 ? 2 : v < 50 ? 3 : v < 75 ? 4 : v < 100 ? 5 : 6;
    const fmt = v => v == null ? '--' : Math.round(v).toLocaleString('en-US');
    const percent = v => v == null ? '--' : `${v.toFixed(1)}%`;
    const params = new URLSearchParams(location.search);
    const requestedHour = params.get('hour');
    const lastScoredIndex = () => hours[hours.length-1].a === null ? hours.length-2 : hours.length-1;
    let index = requestedHour === 'next' ? hours.length - 1 : lastScoredIndex();
    const requestedMeasure = params.get('measure');
    let measure = ['actual','forecast','gap','wape'].includes(requestedMeasure) ? requestedMeasure : 'actual';
    const requestedZone = Number(params.get('zone'));
    let selectedId = byId.has(requestedZone) ? requestedZone : 79, timer = null, activeData = null;
    let historyDuration = ['day','week','month','year'].includes(params.get('duration')) ? params.get('duration') : 'week';
    const slider = document.getElementById('time-slider');
    slider.max = String(hours.length - 1);
    slider.value = String(index);
    const zoneName = id => zoneInfo[byId.get(id)];
    const signalRules = { surgeRatio:2, surgeExtra:40, errorWape:40, errorVolume:150, biasPercent:15, biasVolume:500, biasPersistence:6 };
    function buildSignals() {
      const priorByClock = Array.from({length:24}, () => Array.from({length:zoneInfo.length}, () => []));
      const biasRun = Array(zoneInfo.length).fill(0), biasDirection = Array(zoneInfo.length).fill(0);
      return hours.map((hour,h) => {
        const clock = Number(hour.local.match(/at (\d{2}):/)[1]);
        return zoneInfo.map((zone,z) => {
          const signal = {surge:false,error:false,bias:false,baseline:null,wape:null,volume:null,biasPct:null,biasHours:0};
          if (hour.a === null) return signal;
          const peers = priorByClock[clock][z];
          if (peers.length >= 3) {
            const ordered = [...peers].sort((a,b) => a-b);
            const mid = Math.floor(ordered.length / 2);
            signal.baseline = ordered.length % 2 ? ordered[mid] : (ordered[mid-1] + ordered[mid]) / 2;
            signal.surge = hour.a[z] >= signalRules.surgeRatio * Math.max(10,signal.baseline)
              && hour.a[z] - signal.baseline >= signalRules.surgeExtra;
          }
          peers.push(hour.a[z]);
          if (h >= 23) {
            let actual = 0, forecast = 0, absoluteError = 0;
            for (let k = h - 23; k <= h; k++) {
              actual += hours[k].a[z];
              forecast += hours[k].p[z];
              absoluteError += Math.abs(hours[k].a[z] - hours[k].p[z]);
            }
            signal.volume = actual;
            signal.wape = actual > 0 ? 100 * absoluteError / actual : null;
            signal.biasPct = actual > 0 ? 100 * (forecast - actual) / actual : null;
            signal.error = actual >= signalRules.errorVolume && signal.wape >= signalRules.errorWape;
            const direction = Math.sign(signal.biasPct);
            if (actual >= signalRules.biasVolume && Math.abs(signal.biasPct) >= signalRules.biasPercent) {
              biasRun[z] = direction === biasDirection[z] ? biasRun[z] + 1 : 1;
              biasDirection[z] = direction;
            } else {
              biasRun[z] = 0;
              biasDirection[z] = 0;
            }
            signal.biasHours = biasRun[z];
            signal.bias = biasRun[z] >= signalRules.biasPersistence;
          }
          return signal;
        });
      });
    }
    let signalsAt = buildSignals();
    const signalDuration = (kind,h,z) => {
      let duration = 0;
      while (h >= 0 && signalsAt[h][z][kind]) { duration++; h--; }
      return duration;
    };
    function renderSignalMarkers() {
      const current = signalsAt[index];
      const totals = current.reduce((sum,s) => {
        sum.surge += Number(s.surge); sum.error += Number(s.error); sum.bias += Number(s.bias);
        return sum;
      },{surge:0,error:0,bias:0});
      const candidates = current.map((s,z) => ({s,z})).filter(({s}) => s.surge || s.error || s.bias);
      candidates.sort((a,b) => {
        const rank = ({s,z}) => (s.surge ? 2 + (hours[index].a?.[z] || 0) / Math.max(10,s.baseline || 1) : 0)
          + (s.error ? 2 + s.wape / 100 : 0) + (s.bias ? .5 + Math.abs(s.biasPct) / 100 : 0);
        return rank(b) - rank(a);
      });
      const visible = candidates.slice(0,12);
      const selected = candidates.find(item => zoneInfo[item.z].id === selectedId);
      if (selected && !visible.includes(selected)) visible[visible.length-1] = selected;
      const markerLayer = document.getElementById('signal-markers');
      markerLayer.innerHTML = visible.map(({s,z}) => {
        const box = paths[z].getBBox(), x = box.x + box.width/2, y = box.y + box.height/2;
        const red = s.surge || s.error, color = red ? '#e4514e' : '#c88b3f';
        return `<g transform="translate(${x.toFixed(1)},${y.toFixed(1)})">${red && s.bias ? '<circle r="12" fill="none" stroke="#dfaa60" stroke-width="2"/>' : ''}<circle r="9" fill="${color}" stroke="#111923" stroke-width="2.5"/><text text-anchor="middle" dominant-baseline="central" y=".5" fill="#fff" font-size="12" font-weight="800">!</text></g>`;
      }).join('');
      document.getElementById('signal-map-key').innerHTML = hours[index].a === null
        ? 'Signals appear after actual demand is scored.'
        : `Chart signals &nbsp; <b>!</b> ${totals.surge} surge &middot; ${totals.error} error &nbsp; <em>!</em> ${totals.bias} bias${candidates.length > 12 ? ` &middot; top ${visible.length} shown` : ''}`;
    }
    function trailingWape() {
      const end = Math.min(index, lastScoredIndex());
      if (end < 23) return null;
      const actual = Array(zoneInfo.length).fill(0);
      const error = Array(zoneInfo.length).fill(0);
      for (let h = end - 23; h <= end; h++) {
        for (let z = 0; z < zoneInfo.length; z++) {
          actual[z] += hours[h].a[z];
          error[z] += Math.abs(hours[h].a[z] - hours[h].p[z]);
        }
      }
      const values = actual.map((v,z) => v >= minWapeVolume ? 100 * error[z] / v : null);
      const global = 100 * error.reduce((a,b) => a+b,0) / actual.reduce((a,b) => a+b,0);
      return { values, global, actual, muted: values.filter(v => v === null).length, end };
    }
    function updateLegend() {
      let title, colors, labels;
      if (measure === 'gap') {
        title='Absolute error / hour';
        colors=gapColors;
        labels=['0','1-4','5-14','15-39','40-79','80-149','150+'];
      } else if (measure === 'wape') {
        title='Zone WAPE (24h)';
        colors=wapeColors;
        labels=['<10%','10-19%','20-34%','35-49%','50-74%','75-99%','100%+'];
      } else {
        title='Pickups / hour';
        colors=demandColors;
        labels=['0','1-4','5-14','15-39','40-99','100-199','200+'];
      }
      const legendItems=colors.map((color,i)=>`<span class="legend-item"><i class="swatch" style="background:${color}"></i>${labels[i]}</span>`).join('');
      const muted=measure==='wape' ? `<span class="legend-item"><i class="swatch" style="background:${noDataColor}"></i>&lt;${minWapeVolume} pickups</span>` : '';
      document.getElementById('legend').innerHTML=`<strong>${title}</strong>${legendItems}${muted}`;
    }
    function render() {
      const hour = hours[index];
      if (hour.a === null && (measure === 'actual' || measure === 'gap')) measure = 'forecast';
      if (measure === 'wape' && index < 23) measure = 'actual';
      for (const key of ['actual','forecast','gap','wape']) {
        document.getElementById(`${key}-btn`).classList.toggle('active', measure === key);
      }
      document.getElementById('actual-btn').disabled = hour.a === null;
      document.getElementById('gap-btn').disabled = hour.a === null;
      document.getElementById('wape-btn').disabled = Math.min(index,lastScoredIndex()) < 23;
      const stats = measure === 'wape' ? trailingWape() : null;
      const data = measure === 'actual' ? hour.a
        : measure === 'forecast' ? hour.p
        : measure === 'gap' ? hour.a.map((v,z) => Math.abs(v - hour.p[z]))
        : stats.values;
      activeData = data;
      const palette = measure === 'gap' ? gapColors : measure === 'wape' ? wapeColors : demandColors;
      const bucket = measure === 'gap' ? gapBucket : measure === 'wape' ? wapeBucket : demandBucket;
      for (const path of paths) {
        const id = Number(path.dataset.id), pos = byId.get(id), value = data[pos];
        path.setAttribute('fill', value === null ? noDataColor : palette[bucket(value)]);
        path.classList.toggle('selected', id === selectedId);
      }
      renderSignalMarkers();
      slider.value = String(index);
      const modeNames = { actual:'Observed pickups', forecast:'Model forecast', gap:'Absolute forecast error', wape:'Trailing 24h zone WAPE' };
      const windowEnd = stats ? hours[stats.end].local : hour.local;
      document.getElementById('clock').innerHTML = `${hour.local}<small>${hour.a === null ? 'Next forecast; actual pending' : 'Completed hour; actual available'}</small>`;
      document.getElementById('map-subtitle').textContent = measure === 'wape' ? `24 scored hours through ${windowEnd}` : `${modeNames[measure]} | ${hour.a === null ? 'next hour' : 'completed hour'}`;
      document.getElementById('selected-sub').textContent = `${modeNames[measure]} | ${measure === 'wape' ? windowEnd : hour.local}`;
      document.getElementById('total-label').textContent = measure === 'actual' ? 'Observed citywide pickups'
        : measure === 'forecast' ? 'Forecast citywide pickups'
        : measure === 'gap' ? 'Total absolute error (all zones)'
        : hour.a === null ? 'Citywide WAPE (previous 24 scored hours)' : 'Citywide WAPE (last 24 scored hours)';
      const cityActual = hour.a === null ? null : hour.a.reduce((sum,v) => sum + v,0);
      document.getElementById('total-value').textContent = measure === 'actual' ? fmt(cityActual)
        : measure === 'forecast' ? fmt(hour.pt)
        : measure === 'gap' ? fmt(hour.ae)
        : percent(stats.global);
      document.getElementById('total-foot').textContent = measure === 'wape'
        ? `${stats.muted} low-volume zones muted; fewer than ${minWapeVolume} actual pickups`
        : measure === 'gap' ? 'Sum of absolute zone errors for this hour.'
        : hour.a === null ? 'Actual labels have not arrived at this virtual time.'
        : `${hour.a.filter(v=>v===0).length} zones recorded zero pickups; 263 zones covered`;
      const z = zoneName(selectedId), pos = byId.get(selectedId);
      document.getElementById('zone-title').innerHTML = `${z.name}<small>${z.borough}, TLC zone ${z.id}</small>`;
      document.getElementById('zone-actual').textContent = hour.a === null ? 'Pending' : fmt(hour.a[pos]);
      document.getElementById('zone-forecast').textContent = fmt(hour.p[pos]);
      document.getElementById('zone-third-label').textContent = measure === 'wape' ? (hour.a === null ? 'Previous 24h WAPE' : '24h WAPE') : 'Abs. error';
      document.getElementById('zone-third-label').title = measure === 'wape' ? `Scored hours through ${hours[lastScoredIndex()].local}` : '';
      document.getElementById('zone-third').textContent = measure === 'wape' ? percent(data[pos])
        : hour.a === null ? 'Pending' : fmt(Math.abs(hour.a[pos] - hour.p[pos]));
      const signal = signalsAt[index][pos];
      const signalRows = [];
      if (signal.surge) signalRows.push(`<div class="signal-row"><span class="signal-pill">! SURGE</span><span>${fmt(hour.a[pos])} pickups vs ${fmt(signal.baseline)} same-hour replay median; ${signalDuration('surge',index,pos)}h</span></div>`);
      if (signal.error) signalRows.push(`<div class="signal-row"><span class="signal-pill">! ERROR</span><span>24h WAPE ${percent(signal.wape)} on ${fmt(signal.volume)} pickups; ${signalDuration('error',index,pos)}h</span></div>`);
      if (signal.bias) signalRows.push(`<div class="signal-row"><span class="signal-pill amber">! BIAS WATCH</span><span>${signal.biasPct < 0 ? 'Underforecasting' : 'Overforecasting'} ${percent(Math.abs(signal.biasPct))} over 24h; ${signal.biasHours}h sustained</span></div>`);
      document.getElementById('zone-signals').innerHTML = signalRows.length ? signalRows.join('') : `<span class="signal-none">${hour.a === null ? 'Zone signals await actual demand for this forecast hour.' : 'No signal for this zone at the selected hour.'}</span>`;
      document.getElementById('rank-title').textContent = measure === 'actual' ? 'Highest observed demand'
        : measure === 'forecast' ? 'Highest forecast demand'
        : measure === 'gap' ? 'Largest absolute errors'
        : `Highest WAPE (${minWapeVolume}+ pickups)`;
      const ranking = data.map((v,i) => [v,zoneInfo[i].name]).filter(([v]) => v !== null).sort((a,b)=>b[0]-a[0]).slice(0,5);
      const display = measure === 'wape' ? percent : fmt;
      document.getElementById('top-zones').innerHTML = ranking.map(([v,name]) => `<div class="rank"><span title="${name}">${name}</span><span>${display(v)}</span></div>`).join('');
      document.getElementById('data-note').textContent = measure === 'wape'
        ? `Zone WAPE is total absolute error divided by total actual pickups over 24 complete hours. Zones below ${minWapeVolume} pickups are muted.`
        : measure === 'gap' ? `Color shows the size of the miss. In ${z.name}, the model ${hour.a[pos] >= hour.p[pos] ? 'underestimated' : 'overestimated'} demand by ${fmt(data[pos])} pickups.`
        : hour.a === null ? 'Forecast values are available now. Observed demand appears only after this hour is complete.'
        : 'Color bins stay fixed while you move through time, so changing colors reflect changing pickup counts.';
      updateLegend();
      renderHistory();
    }
    let historySamples = [];
    function zoneRollingWape(hourIndex, pos) {
      if (hourIndex < 23 || hours[hourIndex].a === null) return null;
      let observed = 0, absoluteError = 0;
      for (let h = hourIndex - 23; h <= hourIndex; h++) {
        observed += hours[h].a[pos];
        absoluteError += Math.abs(hours[h].a[pos] - hours[h].p[pos]);
      }
      return observed >= minWapeVolume ? 100 * absoluteError / observed : null;
    }
    const historyCache = new Map();
    async function renderHistory() {
      const z = zoneName(selectedId), pos = byId.get(selectedId);
      const svg = document.getElementById('history-chart');
      const empty = document.getElementById('history-empty');
      const key = document.getElementById('history-key');
      const summary = document.getElementById('history-summary');
      document.getElementById('history-title').textContent = z.name;
      for (const duration of ['day','week','month','year']) {
        document.getElementById(`duration-${duration}`).classList.toggle('active', historyDuration === duration);
      }
      const duration = historyDuration, endHour = hours[index].utc, zoneId = selectedId;
      let samples, start, coverageText = '';
      if (duration === 'month' || duration === 'year') {
        svg.setAttribute('hidden',''); key.hidden = true; summary.hidden = true; empty.hidden = false;
        document.getElementById('history-empty-title').textContent = 'Loading registered history';
        document.getElementById('history-empty-copy').textContent = 'Reading scored zone-hour artifacts.';
        const cacheKey = `${feedVersion}|${zoneId}|${duration}|${endHour}`;
        if (!historyCache.has(cacheKey)) {
          const query = new URLSearchParams({feed:feedMode,zone:String(zoneId),duration,end:endHour});
          historyCache.set(cacheKey, fetch(`/api/history?${query}`,{cache:'no-store'}).then(async response => {
            if (!response.ok) { const problem = await response.json().catch(()=>({})); throw new Error(problem.error || `History HTTP ${response.status}`); }
            return response.json();
          }));
        }
        let result;
        try { result = await historyCache.get(cacheKey); }
        catch (error) {
          historyCache.delete(cacheKey);
          if (duration === historyDuration && zoneId === selectedId && endHour === hours[index].utc) {
            document.getElementById('history-empty-title').textContent = 'History unavailable';
            document.getElementById('history-empty-copy').textContent = error.message;
          }
          return;
        }
        if (duration !== historyDuration || zoneId !== selectedId || endHour !== hours[index].utc) return;
        samples = result.samples;
        if (result.scored_hours < result.expected_hours) coverageText = ` | ${result.scored_hours}/${result.expected_hours} UTC hours scored`;
        if (!samples.length) {
          document.getElementById('history-empty-title').textContent = 'No scored history in this interval';
          document.getElementById('history-empty-copy').textContent = 'Move the replay clock to an hour with registered scores.';
          return;
        }
        start = 0;
      } else {
        const count = duration === 'day' ? 24 : 168;
        start = Math.max(0, index - count + 1);
        samples = [];
        for (let h = start; h <= index; h++) {
          const signal = signalsAt[h][pos];
          samples.push({ hourIndex:h, local:hours[h].local, actual:hours[h].a === null ? null : hours[h].a[pos], forecast:hours[h].p[pos], wape:zoneRollingWape(h,pos), bias:signal.biasPct, signal });
        }
      }
      svg.removeAttribute('hidden');
      key.hidden = false;
      summary.hidden = false;
      empty.hidden = true;
      historySamples = samples;
      const scored = samples.filter(d => d.actual !== null);
      const observed = scored.reduce((sum,d) => sum + d.actual,0);
      const predicted = scored.reduce((sum,d) => sum + d.forecast,0);
      const error = scored.reduce((sum,d) => sum + (d.absolute_error ?? Math.abs(d.actual - d.forecast)),0);
      const periodWape = observed >= minWapeVolume ? percent(100 * error / observed) : 'Low volume';
      document.getElementById('history-sub').textContent = `${z.borough} | TLC zone ${z.id} | ${samples[0].local} to ${samples[samples.length-1].local} | ${duration === 'year' ? 'daily totals' : 'hourly'}${coverageText}`;
      const latestBias = samples[samples.length-1].bias;
      summary.innerHTML = `<span>Observed <strong>${fmt(observed)}</strong></span><span>Forecast for scored hours <strong>${fmt(predicted)}</strong></span><span>Period WAPE <strong>${periodWape}</strong></span><span>Latest ${duration === 'year' ? 'day' : '24h'} signed bias <strong>${latestBias === null ? '--' : `${latestBias >= 0 ? '+' : ''}${percent(latestBias)}`}</strong></span>`;
      const maxDemand = Math.max(1,...samples.map(d => Math.max(d.actual ?? 0,d.forecast)));
      const base = 10 ** Math.floor(Math.log10(maxDemand / 4));
      const step = [1,2,5,10].map(v => v * base).find(v => v * 4 >= maxDemand) || 10 * base;
      const demandMax = step * 4;
      const maxWape = Math.max(20,...samples.map(d => d.wape ?? 0));
      const wapeMax = Math.ceil(maxWape / 20) * 20;
      const maxBias = Math.max(20,...samples.map(d => Math.abs(d.bias ?? 0)));
      const biasMax = Math.ceil(maxBias / 20) * 20;
      const left = 58, right = 1170, demandTop = 40, demandBottom = 270, wapeTop = 308, wapeBottom = 420, biasTop = 460, biasMid = 495, biasBottom = 530;
      const xAt = i => left + i * (right - left) / Math.max(1,samples.length - 1);
      const demandY = v => demandBottom - v / demandMax * (demandBottom - demandTop);
      const wapeY = v => wapeBottom - v / wapeMax * (wapeBottom - wapeTop);
      const biasY = v => biasMid - v / biasMax * (biasMid - biasTop);
      function linePath(field, yFor) {
        let d = '', connected = false;
        samples.forEach((sample,i) => {
          const value = sample[field];
          if (value === null) { connected = false; return; }
          d += `${connected ? 'L' : 'M'}${xAt(i).toFixed(1)},${yFor(value).toFixed(1)} `;
          connected = true;
        });
        return d;
      }
      const demandGrid = [0,1,2,3,4].map(i => {
        const y = demandY(i * step);
        return `<line x1="${left}" y1="${y}" x2="${right}" y2="${y}" stroke="#2b3d4c" stroke-width="1"/><text x="46" y="${y+4}" text-anchor="end" fill="#8399a9" font-size="11">${fmt(i*step)}</text>`;
      }).join('');
      const wapeGrid = [0,.5,1].map(f => {
        const y = wapeY(f * wapeMax);
        return `<line x1="${left}" y1="${y}" x2="${right}" y2="${y}" stroke="#2b3d4c" stroke-width="1"/><text x="46" y="${y+4}" text-anchor="end" fill="#8399a9" font-size="11">${Math.round(f*wapeMax)}%</text>`;
      }).join('');
      const biasGrid = [-1,0,1].map(f => {
        const y = biasY(f * biasMax);
        const label = f > 0 ? `+${biasMax}%` : f < 0 ? `-${biasMax}%` : '0%';
        return `<line x1="${left}" y1="${y}" x2="${right}" y2="${y}" stroke="${f === 0 ? '#607180' : '#2b3d4c'}" stroke-width="1"/><text x="46" y="${y+4}" text-anchor="end" fill="#8399a9" font-size="11">${label}</text>`;
      }).join('');
      const hitWidth = (right-left) / Math.max(1,samples.length-1);
      function signalOverlay(kind, field, yFor, top, bottom, color) {
        let bands = '', strokes = '';
        for (let i = 0; i < samples.length; ) {
          if (!samples[i].signal[kind]) { i++; continue; }
          const first = i;
          while (i+1 < samples.length && samples[i+1].signal[kind]) i++;
          const last = i;
          const x1 = Math.max(left,xAt(first)-hitWidth/2), x2 = Math.min(right,xAt(last)+hitWidth/2);
          bands += `<rect x="${x1}" y="${top}" width="${Math.max(3,x2-x1)}" height="${bottom-top}" fill="${color}" opacity=".12"/>`;
          if (first === last) strokes += `<circle cx="${xAt(first)}" cy="${yFor(samples[first][field])}" r="5" fill="${color}" stroke="#15202a" stroke-width="1.5"/>`;
          else {
            const d = samples.slice(first,last+1).map((sample,j) => `${j ? 'L' : 'M'}${xAt(first+j).toFixed(1)},${yFor(sample[field]).toFixed(1)}`).join(' ');
            strokes += `<path d="${d}" fill="none" stroke="${color}" stroke-width="3.7" stroke-linejoin="round" stroke-linecap="round"/>`;
          }
          i++;
        }
        return {bands,strokes};
      }
      const surgeOverlay = signalOverlay('surge','actual',demandY,demandTop,demandBottom,'#f06d61');
      const errorOverlay = signalOverlay('error','wape',wapeY,wapeTop,wapeBottom,'#f06d61');
      const biasOverlay = signalOverlay('bias','bias',biasY,biasTop,biasBottom,'#e0aa59');
      const tickEvery = duration === 'day' ? 4 : duration === 'week' ? 24 : duration === 'month' ? 96 : 30;
      const ticks = samples.map((sample,i) => {
        if (i % tickEvery && i !== samples.length - 1) return '';
        const label = duration === 'day' ? sample.local.split(' at ')[1].slice(0,5) : duration === 'year' ? sample.local.slice(5) : sample.local.split(',')[0];
        return `<text x="${xAt(i)}" y="582" text-anchor="middle" fill="#8399a9" font-size="11">${label}</text>`;
      }).join('');
      const last = samples[samples.length-1];
      const markers = `<circle cx="${xAt(samples.length-1)}" cy="${demandY(last.forecast)}" r="4" fill="#ffab68" stroke="#15202a" stroke-width="2"/>` +
        (last.actual === null ? '' : `<circle cx="${xAt(samples.length-1)}" cy="${demandY(last.actual)}" r="4" fill="#65c8de" stroke="#15202a" stroke-width="2"/>`);
      const hitTargets = samples.map((sample,i) => `<rect data-history-index="${i}" x="${Math.max(left,xAt(i)-hitWidth/2)}" y="34" width="${hitWidth}" height="499" fill="transparent" style="cursor:crosshair"/>`).join('');
      svg.innerHTML = `<text x="${left}" y="20" fill="#a9becb" font-size="11" font-weight="700" letter-spacing="1.4">PICKUPS / ${duration === 'year' ? 'DAY' : 'HOUR'}</text>` +
        surgeOverlay.bands + demandGrid + `<path d="${linePath('actual',demandY)}" fill="none" stroke="#65c8de" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>` + surgeOverlay.strokes +
        `<path d="${linePath('forecast',demandY)}" fill="none" stroke="#ffab68" stroke-width="2.1" stroke-linejoin="round" stroke-linecap="round" opacity=".93"/>` +
        `<line x1="${left}" y1="285" x2="${right}" y2="285" stroke="#354657"/>` +
        `<text x="${left}" y="299" fill="#a9becb" font-size="11" font-weight="700" letter-spacing="1.4">${duration === 'year' ? 'DAILY' : 'ROLLING 24H'} WAPE</text>` +
        errorOverlay.bands + wapeGrid + `<path d="${linePath('wape',wapeY)}" fill="none" stroke="#b7c976" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>` + errorOverlay.strokes +
        `<line x1="${left}" y1="435" x2="${right}" y2="435" stroke="#354657"/>` +
        `<text x="${left}" y="449" fill="#a9becb" font-size="11" font-weight="700" letter-spacing="1.4">SIGNED ${duration === 'year' ? 'DAILY' : '24H'} BIAS</text>` +
        biasOverlay.bands + biasGrid + `<path d="${linePath('bias',biasY)}" fill="none" stroke="#d7ab64" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>` + biasOverlay.strokes +
        `<line x1="${xAt(samples.length-1)}" y1="34" x2="${xAt(samples.length-1)}" y2="530" stroke="#a7bed0" stroke-dasharray="4 5" opacity=".55"/>` +
        markers + ticks + `<line id="history-hover-guide" x1="0" x2="0" y1="34" y2="530" stroke="#dce8ef" stroke-dasharray="3 4" opacity=".6" visibility="hidden"/>` + hitTargets;
      document.getElementById('history-tooltip').style.display = 'none';
    }
    const historySvg = document.getElementById('history-chart');
    const historyTip = document.getElementById('history-tooltip');
    historySvg.addEventListener('mousemove', e => {
      const target = e.target.closest('[data-history-index]');
      if (!target) return;
      const i = Number(target.dataset.historyIndex), point = historySamples[i];
      if (!point) return;
      const x = 58 + i * (1170 - 58) / Math.max(1,historySamples.length-1);
      const guide = document.getElementById('history-hover-guide');
      guide.setAttribute('x1',x);
      guide.setAttribute('x2',x);
      guide.setAttribute('visibility','visible');
      const signalNames = [point.signal.surge && 'demand surge',point.signal.error && 'error watch',point.signal.bias && 'bias watch'].filter(Boolean);
      historyTip.innerHTML = `<strong>${point.local}</strong><span>Observed: ${point.actual === null ? 'pending' : fmt(point.actual)}</span><span>Forecast: ${fmt(point.forecast)}</span><span>${historyDuration === 'year' ? 'Daily' : '24h'} WAPE: ${point.wape === null ? 'unavailable / low volume' : percent(point.wape)}</span><span>${historyDuration === 'year' ? 'Daily' : '24h'} signed bias: ${point.bias === null ? 'unavailable' : `${point.bias >= 0 ? '+' : ''}${percent(point.bias)}`}</span>${signalNames.length ? `<span style="color:#ffd2c0">Watch: ${signalNames.join(' + ')}</span>` : ''}`;
      const box = document.getElementById('history-chart-wrap').getBoundingClientRect();
      historyTip.style.left = Math.min(e.clientX-box.left+12,box.width-190)+'px';
      historyTip.style.top = Math.max(8,e.clientY-box.top-82)+'px';
      historyTip.style.display = 'block';
    });
    historySvg.addEventListener('mouseleave', () => {
      historyTip.style.display = 'none';
      const guide = document.getElementById('history-hover-guide');
      if (guide) guide.setAttribute('visibility','hidden');
    });
    for (const duration of ['day','week','month','year']) {
      document.getElementById(`duration-${duration}`).addEventListener('click', () => { historyDuration = duration; renderHistory(); });
    }
    const mapArea = document.getElementById('map-area'), tip = document.getElementById('tooltip');
    for (const path of paths) {
      path.addEventListener('mouseenter', () => { const id=Number(path.dataset.id), z=zoneName(id), h=hours[index], pos=byId.get(id), v=activeData[pos]; const metric=measure==='wape' ? `24h WAPE: ${v===null ? 'low volume' : percent(v)}` : measure==='gap' ? `Absolute error: ${fmt(v)}` : `${measure==='actual' ? 'Observed' : 'Forecast'}: ${fmt(v)}`; tip.innerHTML=`<strong>${z.name}</strong><span>${z.borough} · TLC zone ${id}</span><span>${metric}</span><span>Observed: ${h.a === null ? 'pending' : fmt(h.a[pos])} · Forecast: ${fmt(h.p[pos])}</span>`; const watch=signalsAt[index][pos]; const labels=[watch.surge&&'demand surge',watch.error&&'high WAPE',watch.bias&&'bias watch'].filter(Boolean); if(labels.length)tip.innerHTML+=`<span style="color:#ffd2c0;margin-top:4px">Watch: ${labels.join(' + ')}</span>`; tip.style.display='block'; });
      path.addEventListener('mousemove', e => { const b=mapArea.getBoundingClientRect(); tip.style.left=Math.min(e.clientX-b.left+15,b.width-190)+'px'; tip.style.top=Math.max(12,e.clientY-b.top-65)+'px'; });
      path.addEventListener('mouseleave', () => { tip.style.display='none'; });
      path.addEventListener('click', () => { selectedId=Number(path.dataset.id); render(); });
    }
    document.getElementById('actual-btn').addEventListener('click',()=>{ if(hours[index].a!==null){measure='actual';render();} });
    document.getElementById('forecast-btn').addEventListener('click',()=>{measure='forecast';render();});
    document.getElementById('gap-btn').addEventListener('click',()=>{if(hours[index].a!==null){measure='gap';render();}});
    document.getElementById('wape-btn').addEventListener('click',()=>{if(Math.min(index,lastScoredIndex())>=23){measure='wape';render();}});
    document.getElementById('back-btn').addEventListener('click',()=>{index=Math.max(0,index-1);render();});
    document.getElementById('forward-btn').addEventListener('click',()=>{index=Math.min(hours.length-1,index+1);render();});
    slider.addEventListener('input',e=>{index=Number(e.target.value);render();});
    document.getElementById('play-btn').addEventListener('click',e=>{ if(timer){clearInterval(timer);timer=null;e.target.textContent='▶ Play';return;} if(index===hours.length-1)index=0; e.target.textContent='■ Pause'; timer=setInterval(()=>{if(index>=hours.length-1){clearInterval(timer);timer=null;e.target.textContent='▶ Play';return;}index++;render();},280);render();});
    render();
    async function refreshFeed() {
      if (timer) return;
      const statusResponse = await fetch(`/api/status${feedQuery}`, {cache:'no-store'});
      if (!statusResponse.ok) throw new Error(`Feed status returned HTTP ${statusResponse.status}`);
      const status = await statusResponse.json();
      if (status.version === feedVersion) return;
      const selectedHour = hours[index].utc;
      const wasLatest = index === hours.length-1;
      const wasLatestScored = index === lastScoredIndex();
      const response = await fetch(`/api/replay${feedQuery}`, {cache:'no-store'});
      if (!response.ok) throw new Error(`Data feed returned HTTP ${response.status}`);
      const updated = await response.json();
      hours = updated.hours;
      feedVersion = updated.version;
      signalsAt = buildSignals();
      slider.max = String(hours.length-1);
      const sameHour = hours.findIndex(hour => hour.utc === selectedHour);
      index = wasLatest ? hours.length-1 : wasLatestScored ? lastScoredIndex() : Math.max(0,sameHour);
      document.getElementById('feed-badge').textContent = feedMode === 'replay' ? 'Advancing replay' : 'History archive';
      render();
    }
    setInterval(() => refreshFeed().catch(error => {
      document.getElementById('feed-badge').textContent = 'Feed unavailable';
      document.getElementById('feed-badge').title = error.message;
    }), 30000);
    }
    startDashboard().catch(error => {
      document.getElementById('feed-badge').textContent = 'Feed unavailable';
      document.getElementById('clock').textContent = error.message;
      document.getElementById('history-sub').textContent = 'Start the local server with: taxi-mlops serve-dashboard';
    });
