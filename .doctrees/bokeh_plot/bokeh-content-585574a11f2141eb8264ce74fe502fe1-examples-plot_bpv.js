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
    
    
    const element = document.getElementById("00feb812-3a9b-463f-a56a-1278a67c645e");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '00feb812-3a9b-463f-a56a-1278a67c645e' but no matching script tag was found.")
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
                  const docs_json = '{"095fda0e-b6d3-491c-995e-4d3d15a5c99f":{"version":"3.0.2","title":"Bokeh Application","defs":[],"roots":[{"type":"object","name":"GridPlot","id":"p1820","attributes":{"toolbar":{"type":"object","name":"Toolbar","id":"p1819","attributes":{"tools":[{"type":"object","name":"ResetTool","id":"p1740"},{"type":"object","name":"PanTool","id":"p1741"},{"type":"object","name":"BoxZoomTool","id":"p1742","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p1743","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"WheelZoomTool","id":"p1744"},{"type":"object","name":"LassoSelectTool","id":"p1745","attributes":{"renderers":"auto","overlay":{"type":"object","name":"PolyAnnotation","id":"p1746","attributes":{"syncable":false,"level":"overlay","visible":false,"xs":[],"xs_units":"canvas","ys":[],"ys_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"UndoTool","id":"p1747"},{"type":"object","name":"SaveTool","id":"p1818"},{"type":"object","name":"HoverTool","id":"p1749","attributes":{"renderers":"auto"}}]}},"children":[[{"type":"object","name":"Figure","id":"p1709","attributes":{"width":500,"height":500,"x_range":{"type":"object","name":"DataRange1d","id":"p1710"},"y_range":{"type":"object","name":"DataRange1d","id":"p1711"},"x_scale":{"type":"object","name":"LinearScale","id":"p1722"},"y_scale":{"type":"object","name":"LinearScale","id":"p1724"},"title":{"type":"object","name":"Title","id":"p1789","attributes":{"text":"y / y","text_font_size":"14pt"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p1767","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p1761","attributes":{"selected":{"type":"object","name":"Selection","id":"p1762","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p1763"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"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"},"shape":[512],"dtype":"float64","order":"little"}],["y",{"type":"ndarray","array":{"type":"bytes","data":"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"},"shape":[512],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p1768","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p1769"}}},"glyph":{"type":"object","name":"Line","id":"p1764","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#107591"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p1765","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#107591","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p1766","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#107591","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p1777","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p1771","attributes":{"selected":{"type":"object","name":"Selection","id":"p1772","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p1773"},"data":{"type":"map","entries":[["x",[0,1]],["y",[1,1]]]}}},"view":{"type":"object","name":"CDSView","id":"p1778","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p1779"}}},"glyph":{"type":"object","name":"Line","id":"p1774","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"white"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p1775","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"white","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p1776","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"white","line_alpha":0.2}}}},{"type":"object","name":"GlyphRenderer","id":"p1786","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p1780","attributes":{"selected":{"type":"object","name":"Selection","id":"p1781","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p1782"},"data":{"type":"map"}}},"view":{"type":"object","name":"CDSView","id":"p1787","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p1788"}}},"glyph":{"type":"object","name":"Line","id":"p1783","attributes":{"x":{"type":"value","value":0},"y":{"type":"value","value":0},"line_color":"#1f77b4"}},"nonselection_glyph":{"type":"object","name":"Line","id":"p1784","attributes":{"x":{"type":"value","value":0},"y":{"type":"value","value":0},"line_color":"#1f77b4","line_alpha":0.1}},"muted_glyph":{"type":"object","name":"Line","id":"p1785","attributes":{"x":{"type":"value","value":0},"y":{"type":"value","value":0},"line_color":"#1f77b4","line_alpha":0.2}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p1715","attributes":{"tools":[{"id":"p1740"},{"id":"p1741"},{"id":"p1742"},{"id":"p1744"},{"id":"p1745"},{"id":"p1747"},{"type":"object","name":"SaveTool","id":"p1748"},{"id":"p1749"}]}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p1733","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p1736","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p1735"},"major_label_policy":{"type":"object","name":"AllLabels","id":"p1734"}}}],"below":[{"type":"object","name":"LinearAxis","id":"p1726","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p1729","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p1728"},"major_label_policy":{"type":"object","name":"AllLabels","id":"p1727"}}}],"center":[{"type":"object","name":"Grid","id":"p1732","attributes":{"axis":{"id":"p1726"}}},{"type":"object","name":"Grid","id":"p1739","attributes":{"dimension":1,"axis":{"id":"p1733"}}},{"type":"object","name":"BoxAnnotation","id":"p1770","attributes":{"bottom":1.459677659268402,"top":0.6850827603275131,"fill_color":"#107591","fill_alpha":0.1}}],"output_backend":"webgl"}},0,0]]}}]}}';
                  const render_items = [{"docid":"095fda0e-b6d3-491c-995e-4d3d15a5c99f","roots":{"p1820":"00feb812-3a9b-463f-a56a-1278a67c645e"},"root_ids":["p1820"]}];
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