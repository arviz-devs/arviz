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
    
    
    const element = document.getElementById("6712d578-8008-4d03-b2f4-f66ea8e8105f");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '6712d578-8008-4d03-b2f4-f66ea8e8105f' but no matching script tag was found.")
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
                  const docs_json = '{"10b46538-8b7b-4b87-8ecf-ec58781362bf":{"version":"3.0.2","title":"Bokeh Application","defs":[],"roots":[{"type":"object","name":"GridPlot","id":"p2061","attributes":{"toolbar":{"type":"object","name":"Toolbar","id":"p2060","attributes":{"tools":[{"type":"object","name":"ResetTool","id":"p1960"},{"type":"object","name":"PanTool","id":"p1961"},{"type":"object","name":"BoxZoomTool","id":"p1962","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p1963","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"WheelZoomTool","id":"p1964"},{"type":"object","name":"LassoSelectTool","id":"p1965","attributes":{"renderers":"auto","overlay":{"type":"object","name":"PolyAnnotation","id":"p1966","attributes":{"syncable":false,"level":"overlay","visible":false,"xs":[],"xs_units":"canvas","ys":[],"ys_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"UndoTool","id":"p1967"},{"type":"object","name":"SaveTool","id":"p2059"},{"type":"object","name":"HoverTool","id":"p1969","attributes":{"renderers":"auto"}}]}},"children":[[{"type":"object","name":"Figure","id":"p1929","attributes":{"width":500,"height":500,"x_range":{"type":"object","name":"DataRange1d","id":"p1930"},"y_range":{"type":"object","name":"DataRange1d","id":"p1931"},"x_scale":{"type":"object","name":"LinearScale","id":"p1942"},"y_scale":{"type":"object","name":"LinearScale","id":"p1944"},"title":{"type":"object","name":"Title","id":"p2030","attributes":{"text":"y / y","text_font_size":"14pt"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p1987","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p1981","attributes":{"selected":{"type":"object","name":"Selection","id":"p1982","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p1983"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"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"},"shape":[512],"dtype":"float64","order":"little"}],["y",{"type":"ndarray","array":{"type":"bytes","data":"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"},"shape":[512],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p1988","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p1989"}}},"glyph":{"type":"object","name":"Line","id":"p1984","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#107591"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p1985","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#107591","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p1986","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#107591","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p1996","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p1990","attributes":{"selected":{"type":"object","name":"Selection","id":"p1991","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p1992"},"data":{"type":"map"}}},"view":{"type":"object","name":"CDSView","id":"p1997","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p1998"}}},"glyph":{"type":"object","name":"Line","id":"p1993","attributes":{"x":{"type":"value","value":0},"y":{"type":"value","value":0},"line_color":"#1f77b4","line_alpha":0}},"nonselection_glyph":{"type":"object","name":"Line","id":"p1994","attributes":{"x":{"type":"value","value":0},"y":{"type":"value","value":0},"line_color":"#1f77b4","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p1995","attributes":{"x":{"type":"value","value":0},"y":{"type":"value","value":0},"line_color":"#1f77b4","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p2027","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p2021","attributes":{"selected":{"type":"object","name":"Selection","id":"p2022","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p2023"},"data":{"type":"map"}}},"view":{"type":"object","name":"CDSView","id":"p2028","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p2029"}}},"glyph":{"type":"object","name":"Circle","id":"p2024","attributes":{"x":{"type":"value","value":-1.135840820153207},"y":{"type":"value","value":0},"size":{"type":"value","value":6.0},"fill_color":{"type":"value","value":"#107591"}}},"nonselection_glyph":{"type":"object","name":"Circle","id":"p2025","attributes":{"x":{"type":"value","value":-1.135840820153207},"y":{"type":"value","value":0},"size":{"type":"value","value":6.0},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#107591"},"fill_alpha":{"type":"value","value":0.1},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Circle","id":"p2026","attributes":{"x":{"type":"value","value":-1.135840820153207},"y":{"type":"value","value":0},"size":{"type":"value","value":6.0},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#107591"},"fill_alpha":{"type":"value","value":0.2},"hatch_alpha":{"type":"value","value":0.2}}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p1935","attributes":{"tools":[{"id":"p1960"},{"id":"p1961"},{"id":"p1962"},{"id":"p1964"},{"id":"p1965"},{"id":"p1967"},{"type":"object","name":"SaveTool","id":"p1968"},{"id":"p1969"}]}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p1953","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p1956","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p1955"},"major_label_policy":{"type":"object","name":"AllLabels","id":"p1954"}}}],"below":[{"type":"object","name":"LinearAxis","id":"p1946","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p1949","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p1948"},"major_label_policy":{"type":"object","name":"AllLabels","id":"p1947"}}}],"center":[{"type":"object","name":"Grid","id":"p1952","attributes":{"axis":{"id":"p1946"}}},{"type":"object","name":"Grid","id":"p1959","attributes":{"dimension":1,"axis":{"id":"p1953"}}},{"type":"object","name":"Legend","id":"p2019","attributes":{"items":[{"type":"object","name":"LegendItem","id":"p2020","attributes":{"label":{"type":"value","value":"bpv=0.53"},"renderers":[{"id":"p1996"}]}}]}}],"output_backend":"webgl"}},0,0]]}}]}}';
                  const render_items = [{"docid":"10b46538-8b7b-4b87-8ecf-ec58781362bf","roots":{"p2061":"6712d578-8008-4d03-b2f4-f66ea8e8105f"},"root_ids":["p2061"]}];
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