(function() {
  const fn = function() {
    (function(root) {
      function now() {
        return new Date();
      }
    
      const force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
    
    const element = document.getElementById("c6344098-485a-48c1-a117-b27284b9d5a7");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'c6344098-485a-48c1-a117-b27284b9d5a7' but no matching script tag was found.")
        }
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error(url) {
          console.error("failed to load " + url);
        }
    
        for (let i = 0; i < css_urls.length; i++) {
          const url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error.bind(null, url);
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        for (let i = 0; i < js_urls.length; i++) {
          const url = js_urls[i];
          const element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error.bind(null, url);
          element.async = false;
          element.src = url;
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      const js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-gl-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-mathjax-3.0.2.min.js"];
      const css_urls = [];
    
      const inline_js = [    function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        function(Bokeh) {
          (function() {
            const fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                  const docs_json = '{"21c81d42-ef31-4312-b2d4-39f0e441d122":{"version":"3.0.2","title":"Bokeh Application","defs":[],"roots":[{"type":"object","name":"GridPlot","id":"p2296","attributes":{"toolbar":{"type":"object","name":"Toolbar","id":"p2295","attributes":{"tools":[{"type":"object","name":"ResetTool","id":"p2198"},{"type":"object","name":"PanTool","id":"p2199"},{"type":"object","name":"BoxZoomTool","id":"p2200","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p2201","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"WheelZoomTool","id":"p2202"},{"type":"object","name":"LassoSelectTool","id":"p2203","attributes":{"renderers":"auto","overlay":{"type":"object","name":"PolyAnnotation","id":"p2204","attributes":{"syncable":false,"level":"overlay","visible":false,"xs":[],"xs_units":"canvas","ys":[],"ys_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"UndoTool","id":"p2205"},{"type":"object","name":"SaveTool","id":"p2294"},{"type":"object","name":"HoverTool","id":"p2207","attributes":{"renderers":"auto"}}]}},"children":[[{"type":"object","name":"Figure","id":"p2167","attributes":{"width":500,"height":500,"x_range":{"type":"object","name":"DataRange1d","id":"p2168"},"y_range":{"type":"object","name":"DataRange1d","id":"p2169","attributes":{"start":-1.5,"end":0.5}},"x_scale":{"type":"object","name":"LinearScale","id":"p2180"},"y_scale":{"type":"object","name":"LinearScale","id":"p2182"},"title":{"type":"object","name":"Title","id":"p2261","attributes":{"text":"Model comparison\\nhigher is better"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p2227","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p2221","attributes":{"selected":{"type":"object","name":"Selection","id":"p2222","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p2223"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"NP9YMuTHPsA="},"shape":[1],"dtype":"float64","order":"little"}],["y",[-0.75]]]}}},"view":{"type":"object","name":"CDSView","id":"p2228","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p2229"}}},"glyph":{"type":"object","name":"Scatter","id":"p2224","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":6},"line_color":{"type":"value","value":"grey"},"line_width":{"type":"value","value":2},"fill_color":{"type":"value","value":"grey"},"marker":{"type":"value","value":"triangle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p2225","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":6},"line_color":{"type":"value","value":"grey"},"line_alpha":{"type":"value","value":0.1},"line_width":{"type":"value","value":2},"fill_color":{"type":"value","value":"grey"},"fill_alpha":{"type":"value","value":0.1},"hatch_alpha":{"type":"value","value":0.1},"marker":{"type":"value","value":"triangle"}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p2226","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":6},"line_color":{"type":"value","value":"grey"},"line_alpha":{"type":"value","value":0.2},"line_width":{"type":"value","value":2},"fill_color":{"type":"value","value":"grey"},"fill_alpha":{"type":"value","value":0.2},"hatch_alpha":{"type":"value","value":0.2},"marker":{"type":"value","value":"triangle"}}}}},{"type":"object","name":"GlyphRenderer","id":"p2236","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p2230","attributes":{"selected":{"type":"object","name":"Selection","id":"p2231","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p2232"},"data":{"type":"map","entries":[["xs",[[-30.842297204649128,-30.719354305070254]]],["ys",[[-0.75,-0.75]]]]}}},"view":{"type":"object","name":"CDSView","id":"p2237","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p2238"}}},"glyph":{"type":"object","name":"MultiLine","id":"p2233","attributes":{"xs":{"type":"field","field":"xs"},"ys":{"type":"field","field":"ys"},"line_color":{"type":"value","value":"grey"}}},"nonselection_glyph":{"type":"object","name":"MultiLine","id":"p2234","attributes":{"xs":{"type":"field","field":"xs"},"ys":{"type":"field","field":"ys"},"line_color":{"type":"value","value":"grey"},"line_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"MultiLine","id":"p2235","attributes":{"xs":{"type":"field","field":"xs"},"ys":{"type":"field","field":"ys"},"line_color":{"type":"value","value":"grey"},"line_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"GlyphRenderer","id":"p2245","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p2239","attributes":{"selected":{"type":"object","name":"Selection","id":"p2240","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p2241"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"9nkjLIO3PsA0/1gy5Mc+wA=="},"shape":[2],"dtype":"float64","order":"little"}],["y",[0.0,-1.0]]]}}},"view":{"type":"object","name":"CDSView","id":"p2246","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p2247"}}},"glyph":{"type":"object","name":"Circle","id":"p2242","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":6},"line_width":{"type":"value","value":2},"fill_color":{"type":"value","value":null}}},"nonselection_glyph":{"type":"object","name":"Circle","id":"p2243","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":6},"line_alpha":{"type":"value","value":0.1},"line_width":{"type":"value","value":2},"fill_color":{"type":"value","value":null},"fill_alpha":{"type":"value","value":0.1},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Circle","id":"p2244","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":6},"line_alpha":{"type":"value","value":0.2},"line_width":{"type":"value","value":2},"fill_color":{"type":"value","value":null},"fill_alpha":{"type":"value","value":0.2},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"GlyphRenderer","id":"p2254","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p2248","attributes":{"selected":{"type":"object","name":"Selection","id":"p2249","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p2250"},"data":{"type":"map","entries":[["xs",[[-32.0501031513306,-29.383587413132503],[-32.12817382115607,-29.43347768856332]]],["ys",[[0.0,0.0],[-1.0,-1.0]]]]}}},"view":{"type":"object","name":"CDSView","id":"p2255","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p2256"}}},"glyph":{"type":"object","name":"MultiLine","id":"p2251","attributes":{"xs":{"type":"field","field":"xs"},"ys":{"type":"field","field":"ys"}}},"nonselection_glyph":{"type":"object","name":"MultiLine","id":"p2252","attributes":{"xs":{"type":"field","field":"xs"},"ys":{"type":"field","field":"ys"},"line_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"MultiLine","id":"p2253","attributes":{"xs":{"type":"field","field":"xs"},"ys":{"type":"field","field":"ys"},"line_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"Span","id":"p2257","attributes":{"location":-30.716845282231553,"dimension":"height","line_color":"grey","line_width":1.3704965777283857,"line_dash":[6]}}],"toolbar":{"type":"object","name":"Toolbar","id":"p2173","attributes":{"tools":[{"id":"p2198"},{"id":"p2199"},{"id":"p2200"},{"id":"p2202"},{"id":"p2203"},{"id":"p2205"},{"type":"object","name":"SaveTool","id":"p2206"},{"id":"p2207"}]}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p2191","attributes":{"ticker":{"type":"object","name":"FixedTicker","id":"p2219","attributes":{"ticks":[0.0,-1.0],"minor_ticks":[]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p2193"},"axis_label":"ranked models","major_label_overrides":{"type":"map","entries":[[0,"Non-centered 8 schools"],[-0.75,""],[-1,"Centered 8 schools"]]},"major_label_policy":{"type":"object","name":"AllLabels","id":"p2192"}}}],"above":[{"type":"object","name":"Legend","id":"p2258","attributes":{"click_policy":"hide","items":[{"type":"object","name":"LegendItem","id":"p2259","attributes":{"label":{"type":"value","value":"ELPD difference"},"renderers":[{"id":"p2227"},{"id":"p2236"}]}},{"type":"object","name":"LegendItem","id":"p2260","attributes":{"label":{"type":"value","value":"ELPD"},"renderers":[{"id":"p2245"},{"id":"p2254"}]}}]}}],"below":[{"type":"object","name":"LinearAxis","id":"p2184","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p2187","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p2186"},"axis_label":"elpd_loo (log)","major_label_policy":{"type":"object","name":"AllLabels","id":"p2185"}}}],"center":[{"type":"object","name":"Grid","id":"p2190","attributes":{"axis":{"id":"p2184"}}},{"type":"object","name":"Grid","id":"p2197","attributes":{"dimension":1,"axis":{"id":"p2191"}}}],"output_backend":"webgl"}},0,0]]}}]}}';
                  const render_items = [{"docid":"21c81d42-ef31-4312-b2d4-39f0e441d122","roots":{"p2296":"c6344098-485a-48c1-a117-b27284b9d5a7"},"root_ids":["p2296"]}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    let attempts = 0;
                    const timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
    function(Bokeh) {
        }
      ];
    
      function run_inline_js() {
        for (let i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();